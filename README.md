# [NeurIPS 2026] TaxaAdapter: Scaling Fine-grained Species Image Generation To the Tree of Life

[![Project Page](https://img.shields.io/badge/Project-Page-1f72b1?logo=googlechrome&logoColor=white)](https://imageomics.github.io/TaxaAdapter/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.26128-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.26128)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffd21e)](https://huggingface.co/mridulk/TaxaAdapter)

**Abstract:** Generating images across the Tree of Life is difficult: there are over 10M distinct species on Earth, many of which are highly visually similar.
Recent work has made progress on this challenging task by utilizing Linnean taxonomic hierarchy information to relate species, but this alone provides weak visual guidance: it is defined by what is genetically related, not necessarily what is visually similar. We posit that taxonomy is most useful for this task when it is additionally and explicitly aligned with visual similarity. Building on recent Vision Taxonomy Models (VTMs), e.g., BioCLIP and TaxaBind, which provide image-aligned embeddings of taxonomic hierarchy, we propose **TaxaAdapter**, a plug-in that injects VTM-derived embeddings into a frozen text-to-image diffusion model via taxonomy-text dual conditioning. Concretely, we feed taxonomy-image-aligned embeddings as an auxiliary control stream, making synthesis taxonomy-aware while preserving CLIP's flexible text prompting capabilities for attributes such as background, pose, and context. Extensive experiments demonstrate that **TaxaAdapter** consistently improves morphology fidelity and species-identity accuracy over strong baselines, with a cleaner architecture and training recipe. Our results show that coupling diffusion models with pretrained VTMs closes the gap between taxonomic hierarchy and visual similarity between species and offers a practical route to scalable, fine-grained species synthesis. 

---

## Installation
Using Python=3.12

```bash
# Install dependencies
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Training

### Basic Training Command

For training across 4 GPUs

```bash
accelerate launch --num_processes 4 --multi_gpu --mixed_precision "fp16" train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --data_json_file="/path/to/train.json" \
  --data_root_path="/path/to/images" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=64 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/path/to/output" \
  --save_steps=2000 \
  --model_type="bioclip"
```

For training on 1 GPU

```bash
python train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --data_json_file="/path/to/train.json" \
  --data_root_path="/path/to/images" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=64 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/path/to/output" \
  --save_steps=2000 \
  --model_type="bioclip"
```

---

## Converting Checkpoints

After training, convert the checkpoint to adapter weights:

```bash
python checkpoint_bin_change.py --ckpt_dir /path/to/checkpoint-XXXXX
```

This creates:
- `taxa_adapter.bin` - PyTorch format
- `taxa_adapter.safetensors` - Safetensors format

Both formats are compatible with the inference script.

---


## Inference

### Basic Inference Command

```bash
python inference.py \
  --ckpt="/path/to/checkpoint/taxa_adapter.bin" \
  --json_file="/path/to/test.json" \
  --model_type="bioclip" \
  --dataset="inat" \
  --num_samples=10 \
  --steps=50 \
  --guidance_scale=6 \
  --out_dir="/path/to/output"
```

