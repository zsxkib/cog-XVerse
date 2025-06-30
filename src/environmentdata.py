
from huggingface_hub import snapshot_download
import os

# FLUX.1-dev
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    local_dir="./FLUX.1-dev",
    local_dir_use_symlinks=False
)

# Florence-2-large
snapshot_download(
    repo_id="microsoft/Florence-2-large",
    local_dir="./Florence-2-large",
    local_dir_use_symlinks=False
)

# CLIP ViT Large
snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="./clip-vit-large-patch14",
    local_dir_use_symlinks=False
)

# DINO ViT-s16
snapshot_download(
    repo_id="facebook/dino-vits16",
    local_dir="./dino-vits16",
    local_dir_use_symlinks=False
)

# mPLUG Visual Question Answering
snapshot_download(
    repo_id="xingjianleng/mplug_visual-question-answering_coco_large_en",
    local_dir="./mplug_visual-question-answering_coco_large_en",
    local_dir_use_symlinks=False
)

# XVerse
snapshot_download(
    repo_id="ByteDance/XVerse",
    local_dir="./XVerse",
    local_dir_use_symlinks=False
)


os.environ["FLORENCE2_MODEL_PATH"]    = "./checkpoints/Florence-2-large"
os.environ["SAM2_MODEL_PATH"]         = "./checkpoints/sam2.1_hiera_large.pt"
os.environ["FACE_ID_MODEL_PATH"]      = "./checkpoints/model_ir_se50.pth"
os.environ["CLIP_MODEL_PATH"]         = "./checkpoints/clip-vit-large-patch14"
os.environ["FLUX_MODEL_PATH"]         = "./checkpoints/FLUX.1-dev"
os.environ["DPG_VQA_MODEL_PATH"]      = "./checkpoints/mplug_visual-question-answering_coco_large_en"
os.environ["DINO_MODEL_PATH"]         = "./checkpoints/dino-vits16"

