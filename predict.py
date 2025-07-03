# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import tempfile
import time
import uuid
import torch
import numpy as np
import re
import string
import random
import math
from PIL import Image
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download, hf_hub_download, login

# Import the XVerse modules
import src.flux.generate
from src.flux.generate import generate_from_test_sample, seed_everything
from src.flux.pipeline_tools import CustomFluxPipeline, load_modulation_adapter, load_dit_lora
from src.utils.data_utils import get_train_config, image_grid, pil2tensor, json_dump, pad_to_square, cv2pil, merge_bboxes
from eval.tools.face_id import FaceID
from eval.tools.florence_sam import ObjectDetector


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Set up environment variables
        os.environ["XVERSE_PREPROCESSED_DATA"] = f"{os.getcwd()}/proprocess_data"
        
        # Try to use HF token if available
        hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                print("✅ Logged in to Hugging Face")
            except Exception as e:
                print(f"Warning: Could not login to HF: {e}")
        
        # Download required models
        print("Downloading models...")
        
        # FLUX.1-schnell - try to download if not exists locally  
        try:
            if not os.path.exists("./checkpoints/FLUX.1-schnell/model_index.json"):
                print("Downloading FLUX.1-schnell...")
                snapshot_download(
                    repo_id="black-forest-labs/FLUX.1-schnell",
                    local_dir="./checkpoints/FLUX.1-schnell",
                    local_dir_use_symlinks=False
                )
            else:
                print("FLUX.1-schnell already exists locally, skipping download")
        except Exception as e:
            print(f"Warning: Could not download FLUX.1-schnell: {e}")
            print("Attempting to use local model files if available...")

        # Download other models with error handling
        models_to_download = [
            ("microsoft/Florence-2-large", "./checkpoints/Florence-2-large"),
            ("openai/clip-vit-large-patch14", "./checkpoints/clip-vit-large-patch14"),
            ("facebook/dino-vits16", "./checkpoints/dino-vits16"),
            ("xingjianleng/mplug_visual-question-answering_coco_large_en", "./checkpoints/mplug_visual-question-answering_coco_large_en"),
            ("ByteDance/XVerse", "./checkpoints/XVerse"),
        ]
        
        for repo_id, local_dir in models_to_download:
            try:
                if not os.path.exists(f"{local_dir}/config.json") and not os.path.exists(f"{local_dir}/model_index.json"):
                    print(f"Downloading {repo_id}...")
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False
                    )
                else:
                    print(f"{repo_id} already exists locally, skipping download")
            except Exception as e:
                print(f"Warning: Could not download {repo_id}: {e}")
                print("Attempting to use local model files if available...")

        # SAM2.1 model
        try:
            if not os.path.exists("./checkpoints/sam2.1_hiera_large.pt"):
                print("Downloading SAM2.1 model...")
                hf_hub_download(
                    repo_id="facebook/sam2.1-hiera-large",
                    local_dir="./checkpoints/",
                    filename="sam2.1_hiera_large.pt",
                )
            else:
                print("SAM2.1 model already exists locally, skipping download")
        except Exception as e:
            print(f"Warning: Could not download SAM2.1 model: {e}")
            print("Attempting to use local model files if available...")

        # Set model paths
        os.environ["FLORENCE2_MODEL_PATH"] = "./checkpoints/Florence-2-large"
        os.environ["SAM2_MODEL_PATH"] = "./checkpoints/sam2.1_hiera_large.pt"
        os.environ["FACE_ID_MODEL_PATH"] = "./checkpoints/model_ir_se50.pth"
        os.environ["CLIP_MODEL_PATH"] = "./checkpoints/clip-vit-large-patch14"
        os.environ["FLUX_MODEL_PATH"] = "./checkpoints/FLUX.1-schnell"
        os.environ["DPG_VQA_MODEL_PATH"] = "./checkpoints/mplug_visual-question-answering_coco_large_en"
        os.environ["DINO_MODEL_PATH"] = "./checkpoints/dino-vits16"
        
        # Verify face ID model exists (it should be in the repo as Git LFS)
        if not os.path.exists("./checkpoints/model_ir_se50.pth"):
            print("Warning: Face ID model not found at ./checkpoints/model_ir_se50.pth")
            print("This file should be available as a Git LFS file in the repository.")

        self.dtype = torch.bfloat16
        self.device = "cuda"

        # Load config
        config_path = "train/config/XVerse_config_demo.yaml"
        self.config = get_train_config(config_path)
        
        # Configure for inference (disable DiT LoRA to avoid assertion error)
        self.config["model"]["use_dit_lora"] = False
        
        # Initialize models
        print("Initializing models...")
        self.model = CustomFluxPipeline(
            self.config, self.device, torch_dtype=self.dtype,
        )
        self.model.pipe.set_progress_bar_config(leave=False)

        self.face_model = FaceID(self.device)
        self.detector = ObjectDetector(self.device)

        # Load modulation adapter and LoRA
        ckpt_root = "./checkpoints/XVerse"
        self.model.clear_modulation_adapters()
        self.model.pipe.unload_lora_weights()
        
        if not os.path.exists(ckpt_root):
            print("Checkpoint root does not exist.")
        
        modulation_adapter = load_modulation_adapter(
            self.model, self.config, self.dtype, self.device, 
            f"{ckpt_root}/modulation_adapter", is_training=False
        )
        self.model.add_modulation_adapter(modulation_adapter)
        
        if self.config["model"]["use_dit_lora"]:
            load_dit_lora(
                self.model, self.model.pipe, self.config, self.dtype, 
                self.device, f"{ckpt_root}", is_training=False
            )

        print("Setup complete!")

    def resize_keep_aspect_ratio(self, pil_image, target_size=768):
        """Resize image while keeping aspect ratio"""
        H, W = pil_image.height, pil_image.width
        target_area = target_size * target_size
        current_area = H * W
        scaling_factor = (target_area / current_area) ** 0.5
        new_H = int(round(H * scaling_factor))
        new_W = int(round(W * scaling_factor))
        return pil_image.resize((new_W, new_H))

    def auto_caption_image(self, image: Image.Image) -> str:
        """Generate automatic caption for an image"""
        try:
            caption = self.detector.detector.caption(image, "<CAPTION>").strip()
            if caption.endswith("."):
                caption = caption[:-1]
        except Exception as e:
            print(f"Caption generation failed: {e}")
            caption = "an object"
        return caption.lower()

    def det_seg_img(self, image: Image.Image, label: str) -> Image.Image:
        """Detect and segment objects in image based on label"""
        try:
            instance_result_dict = self.detector.get_multiple_instances(
                image, label, min_size=image.size[0]//20
            )
            indices = list(range(len(instance_result_dict["instance_images"])))
            ins, bbox = self.merge_instances(
                image, indices, 
                instance_result_dict["instance_bboxes"], 
                instance_result_dict["instance_images"]
            )
            return ins
        except Exception as e:
            print(f"Object detection/segmentation failed: {e}")
            return image

    def crop_face_img(self, image: Image.Image) -> Image.Image:
        """Crop face from image"""
        try:
            image = pad_to_square(image).resize((2048, 2048))
            face_bbox = self.face_model.detect(
                (pil2tensor(image).unsqueeze(0) * 255).to(torch.uint8).to(self.device), 1.4
            )[0]
            face = image.crop(face_bbox)
            return face
        except Exception as e:
            print(f"Face cropping failed: {e}")
            return image

    def merge_instances(self, orig_img, indices, ins_bboxes, ins_images):
        """Merge multiple object instances into a single image"""
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

    def generate_random_string(self, length=4):
        """Generate random string for temp directories"""
        letters = string.ascii_letters 
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for generation. Use ENT1, ENT2, ENT3 as placeholders for the input subjects.",
            default="ENT1 and ENT2"
        ),
        image1: Path = Input(
            description="First input image", 
            default=None
        ),
        caption1: str = Input(
            description="Caption for first image (leave empty for auto-caption)",
            default=""
        ),
        crop_image1: str = Input(
            description="How to crop first image: 'none', 'face', or 'auto' (crop to caption)",
            default="none",
            choices=["none", "face", "auto"]
        ),
        image2: Path = Input(
            description="Second input image (optional)", 
            default=None
        ),
        caption2: str = Input(
            description="Caption for second image (leave empty for auto-caption)",
            default=""
        ),
        crop_image2: str = Input(
            description="How to crop second image: 'none', 'face', or 'auto' (crop to caption)",
            default="none",
            choices=["none", "face", "auto"]
        ),
        image3: Path = Input(
            description="Third input image (optional)", 
            default=None
        ),
        caption3: str = Input(
            description="Caption for third image (leave empty for auto-caption)",
            default=""
        ),
        crop_image3: str = Input(
            description="How to crop third image: 'none', 'face', or 'auto' (crop to caption)",
            default="none",
            choices=["none", "face", "auto"]
        ),
        use_id1: bool = Input(
            description="Use identity preservation for first image",
            default=True
        ),
        use_id2: bool = Input(
            description="Use identity preservation for second image",
            default=True
        ),
        use_id3: bool = Input(
            description="Use identity preservation for third image",
            default=True
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            default=8,
            ge=4,
            le=40
        ),
        seed: int = Input(
            description="Random seed for generation",
            default=42
        ),
        target_height: int = Input(
            description="Output image height",
            default=768,
            ge=512,
            le=1024
        ),
        target_width: int = Input(
            description="Output image width", 
            default=768,
            ge=512,
            le=1024
        ),
        cond_size: int = Input(
            description="Condition size for generation",
            default=256,
            choices=[256, 384]
        ),
        weight_id: float = Input(
            description="Weight for identity preservation",
            default=1.6,
            ge=0.1,
            le=5.0
        ),
        weight_ip: float = Input(
            description="Weight for IP adaptation",
            default=5.0,
            ge=0.1,
            le=5.0
        ),
        ip_scale: float = Input(
            description="Scale for latent LoRA",
            default=0.85,
            ge=0.5,
            le=1.5
        ),
        vae_lora_scale: float = Input(
            description="Scale for VAE LoRA",
            default=1.3,
            ge=0.5,
            le=1.5
        ),
        vae_skip_start: float = Input(
            description="Start point for VAE skip iteration (0-1)",
            default=0.05,
            ge=0.0,
            le=1.0
        ),
        vae_skip_end: float = Input(
            description="End point for VAE skip iteration (0-1)",
            default=0.8,
            ge=0.0,
            le=1.0
        ),
        double_attention: bool = Input(
            description="Use double attention (advanced setting)",
            default=False
        ),
        single_attention: bool = Input(
            description="Use single attention",
            default=True
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        torch.cuda.empty_cache()
        
        # Prepare session
        session_id = uuid.uuid4().hex
        cur_run_time = time.strftime("%m%d-%H%M%S")
        processed_directory = os.environ["XVERSE_PREPROCESSED_DATA"]
        temp_dir = f"{processed_directory}/{session_id}/{cur_run_time}_{self.generate_random_string(4)}"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Temporary directory created: {temp_dir}")
        
        # Process input images
        images = [image1, image2, image3]
        captions = [caption1, caption2, caption3]
        crop_modes = [crop_image1, crop_image2, crop_image3]
        use_ids = [use_id1, use_id2, use_id3]
        
        # Auto-caption missing captions
        for idx, (img, cap) in enumerate(zip(images, captions)):
            if img is not None:
                if not cap or cap.strip() == "":
                    try:
                        captions[idx] = self.auto_caption_image(Image.open(str(img)).convert("RGB"))
                        print(f"ENT{idx+1} Prompt Missing. Auto Generating Caption")
                    except Exception as e:
                        print(f"Failed to generate caption for image {idx}: {e}")
        
        # Build default prompt if none provided
        if not prompt or prompt.strip() == "":
            ents = [f"ENT{i+1}" for i, img in enumerate(images) if img is not None]
            prompt = " and ".join(ents)
            print(f"Prompt Not Provided, Defaulting to {prompt}")
        
        src_inputs = []
        use_words = []
        
        for i, (image_path, caption, crop_mode) in enumerate(zip(images, captions, crop_modes)):
            if image_path:
                # Load and resize image
                image = Image.open(str(image_path)).convert("RGB")
                
                # Apply cropping if requested
                if crop_mode == "face":
                    image = self.crop_face_img(image)
                elif crop_mode == "auto" and caption.strip():
                    # Use caption for object detection/cropping
                    image = self.det_seg_img(image, caption.strip())
                
                image = self.resize_keep_aspect_ratio(image, 768)
                
                # Prepare word for processing
                if caption.startswith("a ") or caption.startswith("A "):
                    word = caption[2:]
                else:
                    word = caption

                # Case-insensitive replace of the ENT token
                prompt = re.sub(
                    rf"ent{i+1}",    # match "ent1", "ENT1", "Ent1", etc.
                    caption,
                    prompt,
                    flags=re.IGNORECASE
                )
                
                # Save processed image
                save_path = f"{temp_dir}/tmp_resized_input_{i}.png"
                image.save(save_path)
                
                src_inputs.append({
                    "image_path": save_path,
                    "caption": caption
                })
                use_words.append((i, word, word))
        
        # Prepare control weight lambda
        control_weight_lambda = f"0-1:1/{weight_id}/{weight_ip}"
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
                    for is_id in use_ids:
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
        
        # Prepare VAE skip iteration string
        vae_skip_iter = f"0-{vae_skip_start}:1,{vae_skip_end}-1:1"
        
        # Prepare LoRA scale strings
        ip_scale_str = f"0-1:{ip_scale}"
        vae_lora_scale_str = f"0-1:{vae_lora_scale}"
        
        # Prepare test sample
        test_sample = dict(
            input_images=[],
            position_delta=[0, -32],
            prompt=prompt,
            target_height=target_height,
            target_width=target_width,
            seed=seed,
            cond_size=cond_size,
            vae_skip_iter=vae_skip_iter,
            lora_scale=ip_scale_str,
            control_weight_lambda=control_weight_lambda,
            latent_sblora_scale=ip_scale_str,
            condition_sblora_scale=vae_lora_scale_str,
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
        
        # Save test sample
        json_dump(test_sample, f"{temp_dir}/test_sample.json", 'utf-8')
        assert single_attention == True
        target_size = int(round((target_width * target_height) ** 0.5) // 16 * 16)
        
        self.model.config["train"]["dataset"]["val_condition_size"] = cond_size
        self.model.config["train"]["dataset"]["val_target_size"] = target_size
        
        if control_weight_lambda == "no":
            control_weight_lambda = None
        if vae_skip_iter == "no":
            vae_skip_iter = None
        use_condition_sblora_control = True
        use_latent_sblora_control = True
        
        # Generate image
        output_image = generate_from_test_sample(
            test_sample, self.model.pipe, self.model.config, 
            num_inference_steps=num_inference_steps,
            num_images=1, 
            target_height=target_height,
            target_width=target_width,
            seed=seed,
            store_attn_map=False, 
            vae_skip_iter=vae_skip_iter,  
            control_weight_lambda=control_weight_lambda, 
            double_attention=double_attention,  
            single_attention=single_attention,  
            ip_scale=ip_scale_str,
            use_latent_sblora_control=use_latent_sblora_control,
            latent_sblora_scale=ip_scale_str,
            use_condition_sblora_control=use_condition_sblora_control,
            condition_sblora_scale=vae_lora_scale_str,
        )
        
        # Save output
        if isinstance(output_image, list):
            output_image = output_image[0]
        
        output_path = f"/tmp/output_{session_id}.png"
        output_image.save(output_path)
        
        return Path(output_path)
