# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import spaces

import tempfile
from PIL import Image
import gradio as gr
import string
import random, time, math   
import os
import uuid
import src.flux.generate
from src.flux.generate import generate_from_test_sample, seed_everything
from src.flux.pipeline_tools import CustomFluxPipeline, load_modulation_adapter, load_dit_lora
from src.utils.data_utils import get_train_config, image_grid, pil2tensor, json_dump, pad_to_square, cv2pil, merge_bboxes
from eval.tools.face_id import FaceID
from eval.tools.florence_sam import ObjectDetector
import shutil
import yaml
import numpy as np
from huggingface_hub import snapshot_download, hf_hub_download
import torch


os.environ["XVERSE_PREPROCESSED_DATA"] = f"{os.getcwd()}/proprocess_data"


# FLUX.1-schnell
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-schnell",
    local_dir="./checkpoints/FLUX.1-schnell",
    local_dir_use_symlinks=False
)


# Florence-2-large
snapshot_download(
    repo_id="microsoft/Florence-2-large",
    local_dir="./checkpoints/Florence-2-large",
    local_dir_use_symlinks=False
)

# CLIP ViT Large
snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="./checkpoints/clip-vit-large-patch14",
    local_dir_use_symlinks=False
)

# DINO ViT-s16
snapshot_download(
    repo_id="facebook/dino-vits16",
    local_dir="./checkpoints/dino-vits16",
    local_dir_use_symlinks=False
)

# mPLUG Visual Question Answering
snapshot_download(
    repo_id="xingjianleng/mplug_visual-question-answering_coco_large_en",
    local_dir="./checkpoints/mplug_visual-question-answering_coco_large_en",
    local_dir_use_symlinks=False
)

# XVerse
snapshot_download(
    repo_id="ByteDance/XVerse",
    local_dir="./checkpoints/XVerse",
    local_dir_use_symlinks=False
)

hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    local_dir="./checkpoints/",
    filename="sam2.1_hiera_large.pt",
)



os.environ["FLORENCE2_MODEL_PATH"]    = "./checkpoints/Florence-2-large"
os.environ["SAM2_MODEL_PATH"]         = "./checkpoints/sam2.1_hiera_large.pt"
os.environ["FACE_ID_MODEL_PATH"]      = "./checkpoints/model_ir_se50.pth"
os.environ["CLIP_MODEL_PATH"]         = "./checkpoints/clip-vit-large-patch14"
os.environ["FLUX_MODEL_PATH"]         = "./checkpoints/FLUX.1-schnell"
os.environ["DPG_VQA_MODEL_PATH"]      = "./checkpoints/mplug_visual-question-answering_coco_large_en"
os.environ["DINO_MODEL_PATH"]         = "./checkpoints/dino-vits16"

dtype = torch.bfloat16
device = "cuda"

config_path = "train/config/XVerse_config_demo.yaml"

config = config_train = get_train_config(config_path)
# config["model"]["dit_quant"] = "int8-quanto"
config["model"]["use_dit_lora"] = False
model = CustomFluxPipeline(
    config, device, torch_dtype=dtype,
)
model.pipe.set_progress_bar_config(leave=False)

face_model = FaceID(device)
detector = ObjectDetector(device)

config = get_train_config(config_path)
model.config = config

run_mode = "mod_only"
store_attn_map = False
run_name = time.strftime("%m%d-%H%M")

num_inputs = 3

images = []
captions = []
face_btns = []
det_btns = []
vlm_btns = []

idip_checkboxes = []

ckpt_root = "./checkpoints/XVerse"
model.clear_modulation_adapters()
model.pipe.unload_lora_weights()
if not os.path.exists(ckpt_root):
    print("Checkpoint root does not exist.")

modulation_adapter = load_modulation_adapter(model, config, dtype, device, f"{ckpt_root}/modulation_adapter", is_training=False)
model.add_modulation_adapter(modulation_adapter)
if config["model"]["use_dit_lora"]:
    load_dit_lora(model, model.pipe, config, dtype, device, f"{ckpt_root}", is_training=False)

vae_skip_iter = None
attn_skip_iter = 0


def clear_images():
    return [None, ]*num_inputs

@spaces.GPU()
def det_seg_img(image, label):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    instance_result_dict = detector.get_multiple_instances(image, label, min_size=image.size[0]//20)
    indices = list(range(len(instance_result_dict["instance_images"])))
    ins, bbox = merge_instances(image, indices, instance_result_dict["instance_bboxes"], instance_result_dict["instance_images"])
    return ins

@spaces.GPU()
def crop_face_img(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    image = pad_to_square(image).resize((2048, 2048))
    
    face_bbox = face_model.detect(
        (pil2tensor(image).unsqueeze(0) * 255).to(torch.uint8).to(device), 1.4
    )[0]
    face = image.crop(face_bbox)
    return face

@spaces.GPU()
def vlm_img_caption(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    try:
        caption = detector.detector.caption(image, "<CAPTION>").strip()
        if caption.endswith("."):
            caption = caption[:-1]

    except Exception as e:
        print(e)
        caption = ""
    
    caption = caption.lower()
    return caption


def generate_random_string(length=4):
    letters = string.ascii_letters 
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def resize_keep_aspect_ratio(pil_image, target_size=1024):
    H, W = pil_image.height, pil_image.width
    target_area = target_size * target_size
    current_area = H * W
    scaling_factor = (target_area / current_area) ** 0.5  # sqrt(target_area / current_area)
    new_H = int(round(H * scaling_factor))
    new_W = int(round(W * scaling_factor))
    return pil_image.resize((new_W, new_H))


@spaces.GPU()
def generate_image(
    prompt,   
    image_1, caption_1,
    image_2 = None, caption_2 = None,
    image_3 = None, caption_3 = None,
    use_id_1 = True,
    use_id_2 = True,
    use_id_3 = True,
    num_inference_steps = 8,
    cond_size = 256, 
    target_height = 768, 
    target_width = 768, 
    seed = 42, 
    vae_skip_iter = "0-0.05:1,0.8-1:1", 
    control_weight_lambda = "0-1:1/3.5/5",
    double_attention = False,
    single_attention = True,
    ip_scale = "0-1:0.85",
    latent_sblora_scale_str = "0-1:0.85", 
    vae_lora_scale = "0-1:1.3",
    session_id = None,
):

    if session_id is None:
        session_id = uuid.uuid4().hex
    
    torch.cuda.empty_cache()
    num_images = 1

    images = [image_1, image_2, image_3]
    captions = [caption_1, caption_2, caption_3]
    idips_checkboxes = [use_id_1, use_id_2, use_id_3]

    print(f"Length of images: {len(images)}")
    print(f"Length of captions: {len(captions)}")
    
    print(f"Control weight lambda: {control_weight_lambda}")
    if control_weight_lambda != "no":
        parts = control_weight_lambda.split(',')
        new_parts = []
        for part in parts:
            if ':' in part:
                left, right = part.split(':')
                values = right.split('/')
                global_value = values[0]
                id_value = values[1]
                ip_value = values[2]
                new_values = [global_value]
                for is_id in idips_checkboxes:
                    if is_id:
                        new_values.append(id_value)
                    else:
                        new_values.append(ip_value)
                new_part = f"{left}:{('/'.join(new_values))}"
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        control_weight_lambda = ','.join(new_parts)
    
    print(f"Control weight lambda: {control_weight_lambda}")

    src_inputs = []
    use_words = []
    cur_run_time = time.strftime("%m%d-%H%M%S")
    processed_directory = os.environ["XVERSE_PREPROCESSED_DATA"]
    tmp_dir_root = f'{processed_directory}'
    temp_dir = f"{tmp_dir_root}/{session_id}/{cur_run_time}_{generate_random_string(4)}"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temporary directory created: {temp_dir}")
    for i, (image_path, caption) in enumerate(zip(images, captions)):
        if image_path:
            if caption.startswith("a ") or caption.startswith("A "):
                word = caption[2:]
            else:
                word = caption
            
            if f"ENT{i+1}" in prompt:
                prompt = prompt.replace(f"ENT{i+1}", caption)
            
            image = resize_keep_aspect_ratio(Image.open(image_path), 768)
            save_path = f"{temp_dir}/tmp_resized_input_{i}.png"
            image.save(save_path)
            
            input_image_path = save_path

            src_inputs.append(
                {
                    "image_path": input_image_path,
                    "caption": caption
                }
            )
            use_words.append((i, word, word))


    test_sample = dict(
        input_images=[], position_delta=[0, -32], 
        prompt=prompt,
        target_height=target_height,
        target_width=target_width,
        seed=seed,
        cond_size=cond_size,
        vae_skip_iter=vae_skip_iter,
        lora_scale=ip_scale,
        control_weight_lambda=control_weight_lambda,
        latent_sblora_scale=latent_sblora_scale_str,
        condition_sblora_scale=vae_lora_scale,
        double_attention=double_attention,
        single_attention=single_attention,
    )
    if len(src_inputs) > 0:
        test_sample["modulation"] = [
            dict(
                type="adapter",
                src_inputs=src_inputs,
                use_words=use_words,
            ),
        ]
    
    json_dump(test_sample, f"{temp_dir}/test_sample.json", 'utf-8')
    assert single_attention == True
    target_size = int(round((target_width * target_height) ** 0.5) // 16 * 16)
    print(test_sample)

    model.config["train"]["dataset"]["val_condition_size"] = cond_size
    model.config["train"]["dataset"]["val_target_size"] = target_size
    
    if control_weight_lambda == "no":
        control_weight_lambda = None
    if vae_skip_iter == "no":
        vae_skip_iter = None
    use_condition_sblora_control = True
    use_latent_sblora_control = True
    image = generate_from_test_sample(
        test_sample, model.pipe, model.config, 
        num_inference_steps=num_inference_steps
        num_images=num_images, 
        target_height=target_height,
        target_width=target_width,
        seed=seed,
        store_attn_map=store_attn_map, 
        vae_skip_iter=vae_skip_iter,  
        control_weight_lambda=control_weight_lambda, 
        double_attention=double_attention,  
        single_attention=single_attention,  
        ip_scale=ip_scale,
        use_latent_sblora_control=use_latent_sblora_control,
        latent_sblora_scale=latent_sblora_scale_str,
        use_condition_sblora_control=use_condition_sblora_control,
        condition_sblora_scale=vae_lora_scale,
    )
    if isinstance(image, list):
        num_cols = 2
        num_rows = int(math.ceil(num_images / num_cols))
        image = image_grid(image, num_rows, num_cols)

    # save_path = f"{temp_dir}/tmp_result.png"
    # image.save(save_path)

    return image

def create_image_input(index, open=True, indices_state=None):
    accordion_state = gr.State(open)
    with gr.Column():
        with gr.Accordion(f"Input Image {index + 1}", open=accordion_state.value) as accordion:
            image = gr.Image(type="filepath", label=f"Image {index + 1}")
            caption = gr.Textbox(label=f"ENT{index + 1}", value="")
            id_ip_checkbox = gr.Checkbox(value=False, label=f"ID or not {index + 1}", visible=True)
            with gr.Row():
                vlm_btn = gr.Button("Generate Caption")
                face_btn = gr.Button("Crop Face")
                det_btn = gr.Button("Crop to Prompt")
            accordion.expand(
                    inputs=[indices_state],
                    fn = lambda x: update_inputs(True, index, x), 
                    outputs=[indices_state, accordion_state],
                )
            accordion.collapse(
                    inputs=[indices_state],
                    fn = lambda x: update_inputs(False, index, x), 
                    outputs=[indices_state, accordion_state],
                )
    return image, caption, face_btn, det_btn, vlm_btn, accordion_state, accordion, id_ip_checkbox

def create_min_image_input(index, open=True, indices_state=None):

    with gr.Column(min_width=128):
            image = gr.Image(type="filepath", label=f"Image {index + 1}")
            caption = gr.Textbox(label=f"ENT{index + 1} Prompt", value="")
            
            face_btn = gr.Button("Crop to Face")
            det_btn = gr.Button("Crop to Prompt")
        
            id_ip_checkbox = gr.Checkbox(value=True, label=f"ID or not {index + 1}", visible=False)
            with gr.Row():
                vlm_btn = gr.Button("Generate Caption", visible=False)
                
                
    return image, caption, face_btn, det_btn, vlm_btn, id_ip_checkbox


def merge_instances(orig_img, indices, ins_bboxes, ins_images):
    orig_image_width, orig_image_height = orig_img.width, orig_img.height
    final_img = Image.new("RGB", (orig_image_width, orig_image_height), color=(255, 255, 255))
    bboxes = []
    for i in indices:
        bbox = np.array(ins_bboxes[i], dtype=int).tolist()
        bboxes.append(bbox)
        
        img = cv2pil(ins_images[i])
        mask = (np.array(img)[..., :3] != 255).any(axis=-1)
        mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
        final_img.paste(img, (bbox[0], bbox[1]), mask)
    
    bbox = merge_bboxes(bboxes)
    img = final_img.crop(bbox)
    return img, bbox


def change_accordion(at: bool, index: int, state: list):
    print(at, state)
    indices = state
    if at:
        if index not in indices:
            indices.append(index)
    else:
        if index in indices:
            indices.remove(index)
    
    # 确保 indices 是有序的
    indices.sort()
    print(indices)
    return gr.Accordion(open=at), indices

def update_inputs(is_open, index, state: list):
    indices = state
    if is_open:
        if index not in indices:
            indices.append(index)
    else:
        if index in indices:
            indices.remove(index)
    
    indices.sort()
    print(indices)
    return indices, is_open

def start_session(request: gr.Request):
    """
    Initialize a new user session and return the session identifier.
    
    This function is triggered when the Gradio demo loads and creates a unique
    session hash that will be used to organize outputs and temporary files
    for this specific user session.
    
    Args:
        request (gr.Request): Gradio request object containing session information
        
    Returns:
        str: Unique session hash identifier
    """
    return request.session_hash

# Cleanup on unload
def cleanup(request: gr.Request):
    """
    Clean up session-specific directories and temporary files when the user session ends.
    
    This function is triggered when the Gradio demo is unloaded (e.g., when the user
    closes the browser tab or navigates away). It removes all temporary files and
    directories created during the user's session to free up storage space.
    
    Args:
        request (gr.Request): Gradio request object containing session information
    """
    sid = request.session_hash
    if sid:
        d1 = os.path.join(os.environ["XVERSE_PREPROCESSED_DATA"], sid)
        shutil.rmtree(d1, ignore_errors=True)

css = """
#col-container {
    margin: 0 auto;
    max-width: 1400px;
}
"""

if __name__ == "__main__":

    with gr.Blocks(css=css) as demo:
        session_state = gr.State()
        demo.load(start_session, outputs=[session_state])
        indices_state = gr.State([0, 1])

        with gr.Column(elem_id="col-container"):
            gr.Markdown(
            """ # XVerse – Consistent Multi-Subject Control of Identity and Semantic Attributes via DiT Modulation
    
            • Source: [Github](https://github.com/bytedance/XVerse)  
            • HF Space by : [@alexandernasa](https://twitter.com/alexandernasa/)  """
            )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        for i in range(num_inputs):
                            image, caption, face_btn, det_btn, vlm_btn, id_ip_checkbox = create_min_image_input(i, open=i<2, indices_state=indices_state)
                            images.append(image)
                            idip_checkboxes.append(id_ip_checkbox)
                            captions.append(caption)
                            face_btns.append(face_btn)
                            det_btns.append(det_btn)
                            vlm_btns.append(vlm_btn)
                                
                    prompt = gr.Textbox(label="Prompt", placeholder="e.g., ENT1 and ENT2")
                    gen_btn = gr.Button("Generate", variant="primary")
                    steps_slider = gr.Slider(minimum=4, maximum=40, step=8, value=num_inference_steps, label="inference steps")
                    with gr.Accordion("Advanced Settings", open=False):

                        seed = gr.Number(value=42, label="Seed", info="")
                        
                        with gr.Row():
                            target_height = gr.Slider(512, 1024, step=128, value=768, label="Generated Height", info="")
                            target_width = gr.Slider(512, 1024, step=128, value=768, label="Generated Width", info="")
                            cond_size = gr.Slider(256, 384, step=128, value=256, label="Condition Size", info="")
                        with gr.Row():
                            weight_id = gr.Slider(0.1, 5, step=0.1, value=3.5, label="weight_id")
                            weight_ip = gr.Slider(0.1, 5, step=0.1, value=5, label="weight_ip")
                        with gr.Row():
                            ip_scale_str = gr.Slider(0.5, 1.5, step=0.01, value=0.85, label="latent_lora_scale")
                            vae_lora_scale = gr.Slider(0.5, 1.5, step=0.01, value=1.3, label="vae_lora_scale")
                        with gr.Row():
                            vae_skip_iter_s1 = gr.Slider(0, 1, step=0.01, value=0.05, label="vae_skip_iter_before")
                            vae_skip_iter_s2 = gr.Slider(0, 1, step=0.01, value=0.8, label="vae_skip_iter_after")
                        
                
                        with gr.Row():
                            weight_id_ip_str = gr.Textbox(
                                value="0-1:1/3/5",
                                label="weight_id_ip_str",
                                interactive=False, visible=False
                            )
                            weight_id.change(
                                lambda s1, s2: f"0-1:1/{s1}/{s2}",
                                inputs=[weight_id, weight_ip],
                                outputs=weight_id_ip_str
                            )
                            weight_ip.change(
                                lambda s1, s2: f"0-1:1/{s1}/{s2}",
                                inputs=[weight_id, weight_ip],
                                outputs=weight_id_ip_str
                            )
                            vae_skip_iter = gr.Textbox(
                                value="0-0.05:1,0.8-1:1",
                                label="vae_skip_iter",
                                interactive=False, visible=False
                            )
                            vae_skip_iter_s1.change(
                                lambda s1, s2: f"0-{s1}:1,{s2}-1:1",
                                inputs=[vae_skip_iter_s1, vae_skip_iter_s2],
                                outputs=vae_skip_iter
                            )
                            vae_skip_iter_s2.change(
                                lambda s1, s2: f"0-{s1}:1,{s2}-1:1",
                                inputs=[vae_skip_iter_s1, vae_skip_iter_s2],
                                outputs=vae_skip_iter
                            )
                            
                        
                        with gr.Row():
                            db_latent_lora_scale_str = gr.Textbox(
                                value="0-1:0.85",
                                label="db_latent_lora_scale_str",
                                interactive=False, visible=False
                            )
                            sb_latent_lora_scale_str = gr.Textbox(
                                value="0-1:0.85",
                                label="sb_latent_lora_scale_str",
                                interactive=False, visible=False
                            )
                            vae_lora_scale_str = gr.Textbox(
                                value="0-1:1.3",
                                label="vae_lora_scale_str",
                                interactive=False, visible=False
                            )
                            vae_lora_scale.change(
                                    lambda s: f"0-1:{s}",
                                    inputs=vae_lora_scale,
                                    outputs=vae_lora_scale_str
                                )
                            ip_scale_str.change(
                                    lambda s: [f"0-1:{s}", f"0-1:{s}"],
                                    inputs=ip_scale_str,
                                    outputs=[db_latent_lora_scale_str, sb_latent_lora_scale_str]
                                )
            
                        with gr.Row():
                            double_attention = gr.Checkbox(value=False, label="Double Attention", visible=False)
                            single_attention = gr.Checkbox(value=True, label="Single Attention", visible=False)            
        
                        clear_btn = gr.Button("Clear Images")
    
            
                with gr.Column():
                    output = gr.Image(label="Result")
                    
                    examples = gr.Examples(
                        examples=[
                            [
                                "ENT1 with long curly hair wearing ENT2 at Met Gala", 
                                "sample/woman2.jpg", "a woman",
                                "sample/dress.jpg", "a dress",
                            ],
                            [
                                "ENT1 wearing a tiny hat", 
                                "sample/hamster.jpg", "a hamster",
                                None, None
                            ],
                            [
                                "a drawing of ENT1 and ENT2 that the ENT1 is running alongside of a giant ENT2, in style of a comic book", 
                                "sample/woman.jpg", "a woman",
                                "sample/hamster.jpg", "a hamster",
                            ],
                        ],
                        inputs=[
                            prompt, 
                            images[0], captions[0],
                            images[1], captions[1],  
                        ],
                        outputs=output,
                        fn=generate_image,
                        cache_examples=True,
                    )

                    
        
        gen_btn.click(
            generate_image, 
            inputs=[
                prompt, 
                images[0], captions[0], 
                images[1], captions[1], 
                images[2], captions[2], 
                idip_checkboxes[0],
                idip_checkboxes[1],  
                idip_checkboxes[2],  
                steps_slider,
                cond_size, 
                target_height, 
                target_width, 
                seed,
                vae_skip_iter, 
                weight_id_ip_str,
                double_attention, 
                single_attention,
                db_latent_lora_scale_str, 
                sb_latent_lora_scale_str, 
                vae_lora_scale_str,
                session_state,
            ], 
            outputs=output
        )
        clear_btn.click(clear_images, outputs=images)

        for i in range(num_inputs):
            face_btns[i].click(det_seg_img, inputs=[images[i], gr.State("A face")], outputs=[images[i]])
            det_btns[i].click(det_seg_img, inputs=[images[i], captions[i]], outputs=[images[i]])
            vlm_btns[i].click(vlm_img_caption, inputs=[images[i]], outputs=[captions[i]])
            # images[i].upload(vlm_img_caption, inputs=[images[i]], outputs=[captions[i]])

    demo.unload(cleanup)
    demo.queue()
    demo.launch()