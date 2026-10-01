import os
import argparse
from typing import List, Optional
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

import open_clip
from transformers import PretrainedConfig
from rshf.taxabind import TaxaBind

# import your improved class (the one we discussed that supports model_type='bioclip'
# and accepts bioclip tokens / taxa_texts)
from taxa_adapter import IPAdapter
from tqdm import tqdm
import json
from transformers import CLIPTokenizer, CLIPTextModelWithProjection


# ---------- Utilities ----------

def make_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

def make_pipe(base_model: str, vae_model: Optional[str], device: str) -> StableDiffusionPipeline:
    vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=torch.float16) if vae_model else None
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        scheduler=make_scheduler(),
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    ).to(device)
    return pipe

def image_grid(imgs: List[Image.Image], rows: int, cols: int) -> Image.Image:
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, im in enumerate(imgs):
        grid.paste(im, box=((i % cols) * w, (i // cols) * h))
    return grid

def save_images(images: List[Image.Image], out_dir: str, grid_name: str = "grid.png", cols: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(out_dir, f"img_{i:02d}.png"))
    rows = (len(images) + cols - 1) // cols
    grid = image_grid(images, rows, cols)
    grid.save(os.path.join(out_dir, grid_name))

def save_images_per_class(images, save_dir, class_tag):
    """Save images for iNat dataset - original format."""
    os.makedirs(save_dir, exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(save_dir, f"{class_tag}_sample{i:02d}.png"))


def save_images_fishnet(images, save_dir, taxonomic_name, num_samples):
    """Save images for FishNet dataset - uses taxonomic name with underscores.
    
    Args:
        images: List of PIL images
        save_dir: Directory to save images
        taxonomic_name: Full taxonomic name (will be converted to filename)
        num_samples: Number of samples being generated
    """
    os.makedirs(save_dir, exist_ok=True)
    base_filename = taxonomic_name.replace(" ", "_")
    for i, im in enumerate(images):
        if num_samples == 1:
            filename = f"{base_filename}.png"
        else:
            filename = f"{base_filename}_sample_{i+1:03d}.png"
        im.save(os.path.join(save_dir, filename))


def class_dir_from_image_path(rel_path: str) -> str:
    """
    Given e.g.:
      train/04486_Animalia_..._immutabilis/f9f0....jpg
    return:
      04486_Animalia_..._immutabilis
    """
    return os.path.basename(os.path.dirname(rel_path))

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser("TaxaAdapter Inference (BioCLIP)")
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ckpt", required=True, help="Path to taxa_adapter.bin or taxa_adapter.safetensors")

    # BioCLIP inputs
    # p.add_argument("--taxa", nargs="+", required=True,help="One or more taxonomy strings, e.g. 'Animalia Chordata Aves ...'")

    # Generation params
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.5, help="guidance scale")
    p.add_argument("--scale", type=float, default=1.0, help="scale")
    p.add_argument("--num_samples", type=int, default=10, help="images per prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="outputs_bioclip")
    p.add_argument("--num_tokens", type=int, default=4, help="must match training")
    p.add_argument("--prompt", type=str, default=None, help="Prompt for image generation (optional).")
    p.add_argument("--taxonomic_prompt", action="store_true", help="If set, use taxonomic name as prompt.")
    p.add_argument("--levels", type=int, default=7, help="Taxonomic levels to use (1=kingdom,...7=species).")
    

    p.add_argument("--json_file", required=True, help="Path to JSON list of dicts.")
    p.add_argument("--model_type", type=str, default="bioclip", choices=["bioclip", "taxabind", "location", "clip", "taxa_loc_seq_concat", "loc_taxa_seq_concat"], help="Which model type was used during IP-Adapter training?")
    p.add_argument("--dataset", type=str, default="inat", choices=["inat", "fishnet"], help="Dataset type: 'inat' uses folder from image path, 'fishnet' uses taxonomic name")
    return p.parse_args()


# ---------- Main ----------

@torch.inference_mode()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SD pipeline
    pipe = make_pipe(args.base_model, args.vae_model, device)

    # BioCLIP/TaxaBind models and tokenizers
    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    bioclip_tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    taxabind = TaxaBind(config)
    taxabind_image_text_model   = taxabind.get_image_text_encoder().eval()
    taxabind_tokenizer = taxabind.get_tokenizer()   

    if args.model_type == "bioclip":
        tokenizer = bioclip_tok
    elif args.model_type == "taxabind":
        tokenizer = taxabind_tokenizer
    elif args.model_type == "clip":
        clip_ckpt = "openai/clip-vit-large-patch14"
        tokenizer  = CLIPTokenizer.from_pretrained(clip_ckpt)
        clip_text_with_proj = CLIPTextModelWithProjection.from_pretrained(clip_ckpt).eval()

    # Load JSON
    with open(args.json_file, "r") as f:
        items = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    # de-duplicate by taxonomic_name
    seen = set()
    unique_items = []
    for ex in items:
        tax = ex["taxonomic_name"]
        if tax not in seen:
            seen.add(tax)
            unique_items.append(ex)

    print(f"Found {len(unique_items)} unique taxonomic names (from {len(items)} rows).")


    # IP-Adapter wrapper 
    if args.model_type == "clip":
        print("Using CLIP model for IP-Adapter...")
        ip_model = IPAdapter(
            pipe,
            image_encoder_path=None,
            ip_ckpt=args.ckpt,
            device=device,
            model_type=args.model_type,
            bioclip=clip_text_with_proj,
            taxabind=taxabind_image_text_model
        )
    else:
        ip_model = IPAdapter(
            pipe,
            image_encoder_path=None,
            ip_ckpt=args.ckpt,
            device=device,
            model_type=args.model_type,
            bioclip=bioclip_model,
            taxabind=taxabind_image_text_model
        )

    for idx, entry in tqdm(enumerate(unique_items, start=1), total=len(unique_items)):
        taxa_name = entry["taxonomic_name"]

        if args.levels < 7:
            taxa_parts = taxa_name.split(" ")
            taxa_name = " ".join(taxa_parts[: args.levels])
        
        # Determine folder and save strategy based on dataset type
        if args.dataset == "fishnet":
            # For FishNet: use taxonomic_name with spaces replaced by underscores as folder
            folder_name = taxa_name.replace(" ", "_")
            folder_name = folder_name.replace("/", "-")
            save_dir = os.path.join(args.out_dir, folder_name)
        else:
            # For iNat: use folder from image path (original behavior)
            rel_img_path = entry["image_file"]
            class_dir = class_dir_from_image_path(rel_img_path)
            save_dir = os.path.join(args.out_dir, class_dir)

        if args.model_type == "bioclip" or args.model_type == "taxabind":
            tokens = tokenizer(taxa_name).to(device)
        elif args.model_type == "clip":
            tokens = tokenizer(
                taxa_name,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            tokens = tokens.to(device)

        # Generate images
        if args.taxonomic_prompt:
            prompt = 'best quality, high quality photo of ' + taxa_name
        else:
            prompt = args.prompt
        
        images = ip_model.generate(
            pil_image=tokens,
            num_samples=args.num_samples,
            num_inference_steps=args.steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            prompt=prompt,
            scale=args.scale,
        )

        # Save images using appropriate function
        if args.dataset == "fishnet":
            taxa_name = taxa_name.replace("/", "-")
            save_images_fishnet(images, save_dir, taxa_name, args.num_samples)
        else:
            save_images_per_class(images, save_dir, class_dir)
        
        print(f"    Saved {len(images)} images -> {save_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
