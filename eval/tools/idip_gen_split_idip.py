# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
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

import os
import sys
import time
from tqdm import tqdm
import argparse
import math
import random

import torch
import torch.distributed as dist

from src.flux.generate import generate, generate_from_test_sample, seed_everything
from src.flux.pipeline_tools import CustomFluxPipeline, load_modulation_adapter, load_dit_lora
from src.utils.data_utils import get_train_config, get_rank_and_worldsize
from src.utils.data_utils import pad_to_square, pad_to_target, json_dump, json_load, image_grid

import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--condition_size", type=int, default=128)
    parser.add_argument("--save_name", type=str, default="../examples")
    parser.add_argument("--test_list_name", type=str, default="base_test_list_200")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    
    local_rank, global_rank, world_size = get_rank_and_worldsize()
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")
    is_local_main_process = local_rank == 0
    is_main_process = global_rank == 0
    torch.cuda.set_device(local_rank)
    
    dtype = torch.bfloat16
    device = "cuda"
    config_path = args.config_name

    config = get_train_config(config_path)
    config["train"]["dataset"]["val_condition_size"] = args.condition_size
    config["train"]["dataset"]["val_target_size"] = args.target_size
    config["model"]["layer_control"] = False
            
    run_name = time.strftime("%m%d")
    num_images = 4
    ckpt_root = args.model_path
    save_dir = args.save_name

    model = CustomFluxPipeline(config, device, ckpt_root=ckpt_root, torch_dtype=dtype)
    model.pipe.set_progress_bar_config(leave=False)
    model.config = config
    if "py" in args.test_list_name:
        test_list = globals()[args.test_list_name.split("_py")[0]]
        test_list = test_list[5:11] + test_list[17:23] # TODO only for debug
    else:
        test_list = json_load(f"eval/tools/{args.test_list_name}.json", 'utf-8')
    
    num_samples = len(test_list)
    num_ranks = world_size
    assert local_rank == global_rank
    if world_size > 1:
        num_per_rank = math.ceil(num_samples / num_ranks)
        test_list_indices = list(range(num_samples))
        random.seed(0)
        random.shuffle(test_list_indices)
        local_test_list_indices = test_list_indices[local_rank*num_per_rank:(local_rank+1)*num_per_rank]
        print(f"[worker {local_rank}] got {len(local_test_list_indices)} local samples")


    model.clear_modulation_adapters()
    model.pipe.transformer.unload_lora()

    modulation_adapter = load_modulation_adapter(model, config, dtype, device, f"{ckpt_root}/modulation_adapter", is_training=False)
    model.add_modulation_adapter(modulation_adapter)
    if config["model"]["use_dit_lora"]:
        load_dit_lora(model, model.pipe, config, dtype, device, f"{ckpt_root}", is_training=False)

    os.makedirs(save_dir, exist_ok=True)

    # 复制配置文件到 save_dir
    import shutil
    config_dest_path = os.path.join(save_dir, os.path.basename(config_path))
    shutil.copy(config_path, config_dest_path)
    print(f"已复制配置文件到 {config_dest_path}")

    for i in tqdm(local_test_list_indices):
        test_sample = test_list[i]
        prompt_name = test_sample['prompt'][:40].replace(" ","_")
        save_path = f"{save_dir}/{i}_{prompt_name}.png"
        if os.path.exists(save_path):
            print(f"文件 {save_path} 已存在，跳过保存")
            continue
        image = generate_from_test_sample(test_sample, model.pipe, model.config, num_images=num_images, store_attn_map=False, use_idip=True)
        if isinstance(image, list):
            image = image_grid(image, len(image) // 2, 2)
        # print(f"{test_sample['prompt']}")
        image.save(save_path)
        print(f"save results {i} to: {save_path}")
        del image
    del model

if __name__ == "__main__":
    main()
