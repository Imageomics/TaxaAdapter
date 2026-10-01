import os, sys, torch
import argparse
from safetensors.torch import load_file, save_file
from diffusers import UNet2DConditionModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert checkpoint to taxa_adapter.bin and taxa_adapter.safetensors"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing model.safetensors"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model used for training (default: runwayml/stable-diffusion-v1-5)"
    )
    return parser.parse_args()

# Parse arguments
args = parse_args()
BASE_MODEL = args.base_model
CKPT_DIR   = args.ckpt_dir
IN_SFT     = os.path.join(CKPT_DIR, "model.safetensors")
OUT_BIN    = os.path.join(CKPT_DIR, "taxa_adapter.bin")
OUT_SFT    = os.path.join(CKPT_DIR, "taxa_adapter.safetensors")

# Validate checkpoint directory
if not os.path.isdir(CKPT_DIR):
    sys.exit(f"Error: Checkpoint directory does not exist: {CKPT_DIR}")
if not os.path.isfile(IN_SFT):
    sys.exit(f"Error: model.safetensors not found in: {CKPT_DIR}")

print(f"Processing checkpoint: {CKPT_DIR}")
print(f"Base model: {BASE_MODEL}")

# --- Load saved weights ---
sd = load_file(IN_SFT, device="cpu")

# projector params live under image_proj_model.*
image_proj_sd = {k.replace("image_proj_model.", "", 1): v
                 for k, v in sd.items() if k.startswith("image_proj_model.")}
print("projector param tensors:", len(image_proj_sd))  # expect 4 (proj/norm {weight,bias})

# Build a reference UNet to get the exact attn processor order (so indices match)
unet_ref = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
names_ordered = list(unet_ref.attn_processors.keys())
bases = [n[:-len(".processor")] if n.endswith(".processor") else n for n in names_ordered]

taxa_adapter_sd = {}
missing_cnt = 0
for idx, base in enumerate(bases):
    k_key = f"unet.{base}.processor.to_k_ip.weight"
    v_key = f"unet.{base}.processor.to_v_ip.weight"
    if k_key in sd and v_key in sd:
        taxa_adapter_sd[f"{idx}.to_k_ip.weight"] = sd[k_key]
        taxa_adapter_sd[f"{idx}.to_v_ip.weight"] = sd[v_key]
    else:
        missing_cnt += 1

print("taxa-adapter param tensors:", len(taxa_adapter_sd), "(layers without IP weights:", missing_cnt, ")")
if not image_proj_sd or not taxa_adapter_sd:
    # help you debug prefixes quickly
    print("\nExample keys containing 'processor' or 'image_proj_model':")
    shown = 0
    for k in sd.keys():
        if "processor" in k or "image_proj_model" in k:
            print("  ", k)
            shown += 1
            if shown >= 30: break
    sys.exit("No params found; adjust prefixes or BASE_MODEL/CKPT_DIR.")

# --- Save nested .bin (your loader's non-safetensors path) ---
torch.save({"image_proj": image_proj_sd, "taxa_adapter": taxa_adapter_sd}, OUT_BIN)
print(f"✓ Wrote {OUT_BIN}")

# --- Also save flat .safetensors (your loader's safetensors path) ---
flat = {f"image_proj.{k}": v for k, v in image_proj_sd.items()}
flat.update({f"taxa_adapter.{k}": v for k, v in taxa_adapter_sd.items()})
save_file(flat, OUT_SFT)
print(f"✓ Wrote {OUT_SFT}")

print(f"\n{'='*60}")
print(f"Conversion complete!")
print(f"{'='*60}")
print(f"Output files:")
print(f"  - {OUT_BIN}")
print(f"  - {OUT_SFT}")
print(f"{'='*60}")
