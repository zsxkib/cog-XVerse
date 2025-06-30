# XVerse: Consistent Multi-Subject Control of Identity and Semantic Attributes via DiT Modulation

<p align="center">
    <a href="https://arxiv.org/abs/2506.21416">
            <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2506.21416-b31b1b.svg">
    </a>
    <a href="https://bytedance.github.io/XVerse/">
        <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue">
    </a>
    <a href="https://github.com/bytedance/XVerse/tree/main/assets">
        <img alt="Build" src="https://img.shields.io/badge/XVerseBench-Dataset-green">
    </a>
    <a href="https://huggingface.co/ByteDance/XVerse">
        <img alt="Build" src="https://img.shields.io/badge/🤗-HF%20Model-yellow">
    </a>    
</p>

## 🔥 News
- **2025.6.26**: The code has been released!

![XVerse's capability in single/multi-subject personalization and semantic attribute control (pose, style, lighting)](sample/first_page.png)

## 📖 Introduction

**XVerse** introduces a novel approach to multi-subject image synthesis, offering **precise and independent control over individual subjects** without disrupting the overall image latents or features. We achieve this by transforming reference images into offsets for token-specific text-stream modulation.

This innovation enables high-fidelity, editable image generation where you can robustly control both **individual subject characteristics** (identity) and their **semantic attributes**. XVerse significantly enhances capabilities for personalized and complex scene generation.

## ⚡️ Quick Start

### Requirements and Installation

First, install the necessary dependencies:

```bash
# Create a conda environment named XVerse with Python version 3.10.16
conda create -n XVerse python=3.10.16 -y
# Activate the XVerse environment
conda activate XVerse
# Use pip to install the dependencies specified in requirements.txt
pip install -r requirements.txt
```

Next, download the required checkpoints:
```bash
cd checkpoints
bash ./download_ckpts.sh
cd ..
```
**Important**: You'll also need to download the face recognition model `model_ir_se50.pth` from [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) and place it directly into the `./checkpoints/` folder.

### Local Gradio Demo

To run the interactive Gradio demo locally, execute the following command:
```bash
bash run_demo.sh
```

#### Input Settings Explained
The Gradio demo provides several parameters to control your image generation process:
* **Prompt**: The textual description guiding the image generation.
* **Generated Height/Width**: Use the sliders to set the shape of the output image.
* **Weight_id/ip**: Adjust these weight parameters. Higher values generally lead to better subject consistency but might slightly impact the naturalness of the generated image.
* **latent_lora_scale and vae_lora_scale**: Control the LoRA scale. Similar to Weight_id/ip, larger LoRA values can improve subject consistency but may reduce image naturalness.
* **vae_skip_iter_before and vae_skip_iter_after**: Configure VAE skip iterations. Skipping more steps can result in better naturalness but might compromise subject consistency.

#### Input Images

The demo provides detailed control over your input images:

* **Expand Panel**: Click "Input Image X" to reveal the options for each image.
* **Upload Image**: Click "Image X" to upload your desired reference image.
* **Image Description**: Enter a description in the "Caption X" input box. You can also click "Auto Caption" to generate a description automatically.
* **Detection & Segmentation**: Click "Det & Seg" to perform detection and segmentation on the uploaded image.
* **Crop Face**: Use "Crop Face" to automatically crop the face from the image.
* **ID Checkbox**: Check or uncheck "ID or not" to determine whether to use ID-related weights for that specific input image.

> **⚠️ Important Usage Notes:**
>
> * **Prompt Construction**: The main text prompt **MUST** include the exact text you entered in the `Image Description` field for each active image. **Generation will fail if this description is missing from the prompt.**
>     * *Example*: If you upload two images and set their descriptions as "a man with red hair" (for Image 1) and "a woman with blue eyes" (for Image 2), your main prompt might be: "A `a man with red hair` walking beside `a woman with blue eyes` in a park."
>     * You can then write your main prompt simply as: "`ENT1` walking beside `ENT2` in a park." The code will **automatically replace** these placeholders with the full description text before generation.
> * **Active Images**: Only images in **expanded** (un-collapsed) panels will be fed into the model. Collapsed image panels are ignored.

## Inference with XVerseBench

![XVerseBench](sample/XVerseBench.png)

First, please download XVerseBench according to the contents in the `assets` folder. Then, when running inference, please execute the following command:
```bash
bash ./eval/eval_scripts/run_eval.sh
```
The script will automatically evaluate the model on the XVerseBench dataset and save the results in the `./results` folder.

## 📌 ToDo

- [x] Release github repo.
- [x] Release arXiv paper.
- [x] Release model checkpoints.
- [x] Release inference data: XVerseBench.
- [x] Release inference code for XVerseBench.
- [x] Release inference code for gradio demo.
- [ ] Release inference code for single sample.
- [ ] Release huggingface space demo.
- [ ] Release Benchmark Leaderboard.

## License
    
The code in this project is licensed under Apache 2.0; the dataset is licensed under CC0, subject to the intellctual property owned by Bytedance. Meanwhile, the dataset is adapted from [dreambench++](https://dreambenchplus.github.io/), you should also comply with the license of dreambench++.
    
##  Citation
If XVerse is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our paper:
```bibtex
@article{chen2025xverse,
  title={XVerse: Consistent Multi-Subject Control of Identity and Semantic Attributes via DiT Modulation},
  author={Chen, Bowen and Zhao, Mengyi and Sun, Haomiao and Chen, Li and Wang, Xu and Du, Kang and Wu, Xinglong},
  journal={arXiv preprint arXiv:2506.21416},
  year={2025}
}
```
