# Apertus Recipes

Fine-tuning examples for Swiss AI’s **Apertus language models** (8B and 70B).

---

## 🔗 Resources
- [Apertus 8B Instruct](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)  
- [Apertus 70B Instruct](https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509)  
- [Full collection on HF](https://huggingface.co/collections/swiss-ai/apertus-llm-68b699e65415c231ace3b059)  

---

## ⚡ Quickstart

```bash
# 1. Create and activate environment
uv venv apertus --python 3.10 && source apertus/bin/activate

# 2. Install PyTorch (CUDA 12.8 wheels)
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu128

# 3. Install project requirements
uv pip install -r requirements.txt

# 4. Launch LoRA training on a single GPU
python sft_train.py --config configs/sft_lora.yaml
````

---

## ⚙️ Model Selection

All scripts work with both 8B and 70B versions. Switch model size by editing `model_path`:

```python
# Default: 8B
model_path = "swiss-ai/Apertus-8B-Instruct-2509"

# To use 70B:
# model_path = "swiss-ai/Apertus-70B-Instruct-2509"
```

Device mapping and configuration are handled automatically.

---

## 🏋️ Fine-Tuning

### Full-parameter training (4 GPUs)

```bash
# Standard attention
accelerate launch --config_file configs/zero3.yaml sft_train.py --config configs/sft_full.yaml

# With FlashAttention3
accelerate launch --config_file configs/zero3.yaml sft_train.py \
    --config configs/sft_full.yaml \
    --attn_implementation kernels-community/vllm-flash-attn3
```

### LoRA training (1 GPU)

```bash
python sft_train.py --config configs/sft_lora.yaml
```

---

## 🔧 Customization

You can adjust datasets and hyperparameters either by editing the config YAMLs (`sft_lora.yaml`, `sft_full.yaml`) or passing overrides directly:

```bash
accelerate launch --config_file configs/zero3.yaml \
    sft_train.py --config configs/sft_full.yaml \
    --dataset_name YOUR_DATASET
```

---

