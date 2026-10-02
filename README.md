# Apertus Fine-Tuning Recipes

This repository provides fine-tuning recipes for the [Apertus models](https://apertus-ai.org) from the [Swiss AI Initiative](https://swiss-ai.org), supporting both full-parameter and LoRA-based approaches.

For more background, please see our [online documentation](https://apertus-ai.org/docs/tech/).

Built on top of popular frameworks including TRL, Accelerate, and Transformers, the recipes are optimized for efficient training on modern GPUs, and cover several techniques. LoRA fine-tuning of the smaller models can be done on a single GPU, while full-parameter training the 70B model typically requires a multi-GPU setup and considerable time.


## 🔗 Resources

- [Apertus 1.5](https://huggingface.co/collections/swiss-ai/apertus-v15) (Latest)
- [Apertus 1.1 Mini](https://huggingface.co/collections/swiss-ai/apertus-mini)
- [Apertus 1.0](https://huggingface.co/collections/swiss-ai/apertus-llm-68b699e65415c231ace3b059)

---

## ⚡ Quickstart

```bash
# 1. Create and activate environment (Python 3.12)
uv venv apertus --python 3.12 && source apertus/bin/activate

# 2. Install PyTorch (see link below)
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu # CPU-only
# --index-url https://download.pytorch.org/whl/test/cu128 # CUDA 12.8

# 3. Install project requirements
uv pip install -r requirements.txt

# 4. Launch LoRA training on a single GPU
python sft_train.py --config configs/sft_lora.yaml
```

See the [Astral](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) or [PyTorch](https://pytorch.org/get-started/locally/) docs for details on installing this library, as appropriate for your hardware.

---

## Model Selection

Our scripts work with all versions of the Apertus model family. 
Switch model size by editing the top of `configs/*.yaml`:

```yaml
# Default: v1.5 8B
model_name_or_path: swiss-ai/Apertus-v1.5-8B
```

Device mapping and configuration are handled automatically.

---

## Fine-Tuning

### LoRA training (1 GPU)

```bash
python sft_train.py --config configs/sft_lora.yaml
```

---

### Full-parameter training (4 GPUs)

```bash
# Standard attention
accelerate launch --config_file configs/zero3.yaml sft_train.py --config configs/sft_full.yaml

# With FlashAttention3
accelerate launch --config_file configs/zero3.yaml sft_train.py \
    --config configs/sft_full.yaml \
    --attn_implementation kernels-community/vllm-flash-attn3
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

LoRA adapters are lightweight, and can be loaded together with the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_type = "swiss-ai/Apertus-v1.5-8B"

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(model_type)
tokenizer = AutoTokenizer.from_pretrained(model_type)

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
- [Oleg Lavrovsky](https://people.epfl.ch/oleg.lavrovsky)
