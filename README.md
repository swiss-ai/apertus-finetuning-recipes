# Apertus Fine-Tuning Recipes

This repository provides fine-tuning recipes for Swiss AI’s Apertus language models (8B and 70B), supporting both full-parameter and LoRA-based approaches.
Built on top of popular frameworks including TRL, Accelerate, and Transformers, the recipes are optimized for efficient training on modern GPUs.
LoRA fine-tuning of the 8B model can be done on a single 40 GB GPU, while training the 70B model requires a multi-GPU setup.


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

## Model Selection

All scripts work with both 8B and 70B versions. Switch model size by editing `model_path`:

```python
# Default: 8B
model_path = "swiss-ai/Apertus-8B-Instruct-2509"

# To use 70B:
# model_path = "swiss-ai/Apertus-70B-Instruct-2509"
```

Device mapping and configuration are handled automatically.

---

## Fine-Tuning

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

### Multi-Node training (3 nodes x 4 GPUs)

```bash
# Standard attention
bash --nodes=3 submit_multinode.sh
```
## Customization

You can adjust datasets and hyperparameters either by editing the config YAMLs (`sft_lora.yaml`, `sft_full.yaml`) or passing overrides directly:

```bash
accelerate launch --config_file configs/zero3.yaml \
    sft_train.py --config configs/sft_full.yaml \
    --dataset_name YOUR_DATASET
```

---

## Model Saving


After training completes, your fine-tuned models are saved in the following locations:

- **LoRA Training**: `Apertus-FT/output/apertus_lora/`
- **Full Fine-tuning**: `Apertus-FT/output/apertus_full/`

Each output directory contains:
- `adapter_model.safetensors` (LoRA only) - The LoRA adapter weights
- `adapter_config.json` (LoRA only) - LoRA configuration
- `training_args.bin` - Training arguments used
- `trainer_state.json` - Training state and metrics
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `config.json` - Model configuration

---

## Using Your Fine-tuned Models

#### For LoRA Adapters

LoRA adapters are lightweight and can be loaded with the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")
tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "Apertus-FT/output/apertus_lora/")

# For inference, you can merge the adapter (optional)
model = model.merge_and_unload()
```

#### For Full Fine-tuned Models

Full fine-tuned models can be loaded directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("Apertus-FT/output/apertus_full/")
tokenizer = AutoTokenizer.from_pretrained("Apertus-FT/output/apertus_full/")
```

---

## Contributors

- [Kaustubh Ponkshe](https://kaustubhp11.github.io/)
- [Raghav Singhal](https://raghavsinghal10.github.io/)
