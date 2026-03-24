# =========================
# GLOBAL IMPORTS
# =========================
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os
import json
import matplotlib.pyplot as plt

from transformers import (
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
    BlipProcessor,
    BlipForConditionalGeneration
)

from diffusers import StableDiffusionImg2ImgPipeline
from controlnet_aux import MidasDetector


# =========================
# SKETCH → COMBINED MAP
# =========================
def sketch_to_combined_map(sketch, device="cuda"):

    WIDTH, HEIGHT = 256, 256
    sketch = sketch.convert("RGB").resize((WIDTH, HEIGHT))

    # EDGE
    edges = cv2.Canny(np.array(sketch), 100, 200)
    edges = np.stack([edges]*3, axis=-1)
    edge_image = Image.fromarray(edges)

    # DEPTH
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
    depth_image = midas(sketch).convert("RGB")

    # SEGMENTATION
    processor_seg = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    model_seg = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small"
    ).to(device)

    inputs = processor_seg(images=sketch, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model_seg(**inputs)

    seg = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label in np.unique(seg):
        seg_color[seg == label] = np.random.randint(0, 255, 3)

    seg_image = Image.fromarray(seg_color)

    # COMBINE
    edge_np = np.array(edge_image) / 255.0
    depth_np = np.array(depth_image) / 255.0
    seg_np = np.array(seg_image) / 255.0

    combined = 0.5 * edge_np + 0.3 * depth_np + 0.2 * seg_np
    combined = np.clip(combined * 1.5, 0, 1)
    combined = cv2.GaussianBlur(combined, (5,5), 0)

    combined = (combined * 255).astype(np.uint8)

    return Image.fromarray(combined).resize((512, 512))


# =========================
# SKETCH → PROMPT
# =========================
def generate_prompt_from_sketch(sketch, device="cuda"):

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    inputs = processor(sketch, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return f"{caption}, realistic, 4k, highly detailed"


# =========================
# MRU
# =========================
def run_mru_on_combined(combined_image, mru_path, device="cuda"):

    class MRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.dec = torch.nn.Conv2d(4, 3, 3, padding=1)

        def forward(self, x):
            latent = self.enc(x)
            refined = self.dec(latent)
            return refined, latent

    model = MRU().to(device)
    model.load_state_dict(torch.load(mru_path, map_location=device))
    model.eval()

    img = np.array(combined_image).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        refined, latent = model(img)

    refined = refined.squeeze().permute(1, 2, 0).cpu().numpy()
    refined = (refined * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(refined)


# =========================
# DIFFUSION
# =========================
def run_diffusion(combined_image, prompt, device="cuda"):

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    pipe.enable_attention_slicing()

    image = np.array(combined_image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device).half()
    image = 2 * image - 1

    with torch.no_grad():
        latents = pipe.vae.encode(image).latent_dist.sample()

    latents = latents * 0.18215
    latents = latents.half()

    result = pipe(
        prompt=prompt,
        image=combined_image,
        latents=latents,
        strength=0.6,
        guidance_scale=9.0,
        num_inference_steps=30
    )

    return result.images[0]


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--sketch', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    config = args()

    if config.seed is not None:
        torch.manual_seed(config.seed)

    # -------------------------
    # LOAD SKETCH
    # -------------------------
    sketch = Image.open(config.sketch).convert("RGB")

    # -------------------------
    # STEP 1: COMBINED MAP
    # -------------------------
    combined = sketch_to_combined_map(sketch)

    # -------------------------
    # STEP 2: PROMPT
    # -------------------------
    if config.prompt is None:
        prompt = generate_prompt_from_sketch(sketch)
    else:
        prompt = config.prompt

    print("Prompt:", prompt)

    # -------------------------
    # STEP 3: DIFFUSION
    # -------------------------
    output = run_diffusion(combined, prompt)

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    fname = os.path.splitext(os.path.basename(config.sketch))[0]

    if not os.path.isdir(config.output_dir):
        os.makedirs(config.output_dir)

    image_path = os.path.join(config.output_dir, fname + '_output.png')
    config_path = os.path.join(config.output_dir, fname + '_config.json')

    output.save(image_path)

    with open(config_path, 'w') as fp:
        json.dump(vars(config), fp, indent=2)

    print("Saved:", image_path)

    # -------------------------
    # SHOW SIDE BY SIDE
    # -------------------------
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(sketch)
    plt.title("Sketch")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(output)
    plt.title("Generated Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()