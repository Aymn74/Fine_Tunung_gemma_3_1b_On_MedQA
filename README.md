# MedQA Gemma 1B LoRA Fine-Tuning

Fine-tuning workflow for **Gemma 3 1B** on the **MedQA-USMLE 4-options** dataset using **Unsloth**, **TRL**, and **LoRA**.

This repository keeps the project in a script-first format so it is easier to run, review, and version than a notebook-only Colab export.

## Overview

The project demonstrates a lightweight medical LLM fine-tuning pipeline:

- Load the MedQA-USMLE multiple-choice dataset from Hugging Face
- Convert each example into Gemma chat-style instruction text
- Apply parameter-efficient LoRA adapters
- Fine-tune Gemma 3 1B with Unsloth and TRL
- Run a simple medical QA inference check
- Save the trained LoRA adapter locally

## Model and Dataset

| Item | Value |
|---|---|
| Base Model | `unsloth/gemma-3-1b-it` |
| Dataset | `GBaker/MedQA-USMLE-4-options` |
| Fine-tuning Method | LoRA |
| Framework | Unsloth + TRL |
| Training Samples | 500 |
| Max Steps | 30 |
| GPU Used | Tesla T4 |
| Platform Used | Google Colab |

The trained LoRA adapter is available on Hugging Face:

https://huggingface.co/Ay74/medqa-gemma-1b-lora

## Repository Contents

```text
.
|-- fine_tuning_gemma_1b_medqa.py  # Training and inference script
|-- requirements.txt               # Python dependencies
|-- README.md                      # Project documentation
|-- LICENSE                        # Apache License 2.0
`-- .gitignore                     # Ignored caches, checkpoints, and model outputs
```

## Installation

Use a CUDA GPU runtime. The original experiment was run on Google Colab with a Tesla T4.

```bash
pip install -r requirements.txt
```

If you are running in a fresh Colab notebook, the equivalent direct install command is:

```bash
pip install unsloth datasets transformers trl peft accelerate bitsandbytes torch
```

## Usage

Run the training script:

```bash
python fine_tuning_gemma_1b_medqa.py
```

The script will:

1. Check CUDA/GPU availability
2. Load `unsloth/gemma-3-1b-it` in 4-bit mode
3. Load and format 500 MedQA training samples
4. Apply LoRA adapters
5. Fine-tune for 30 steps
6. Run a sample inference prompt
7. Save the LoRA adapter to `medqa_gemma_lora/`

## Example Inference

Prompt summary:

```text
A pregnant woman has dysuria without fever or flank pain.
What is the best treatment?
```

Expected answer:

```text
B. Nitrofurantoin
```

## Outputs

Generated training artifacts are intentionally excluded from Git:

- `outputs/`
- `checkpoints/`
- `medqa_gemma_lora/`
- model weight files such as `.safetensors`, `.bin`, `.pt`, and `.pth`

This keeps the repository lightweight while allowing the trained adapter to be published separately on Hugging Face.

## Project Purpose

This project was created as an experimental healthcare AI workflow to explore:

- Medical instruction tuning
- Domain adaptation
- Parameter-efficient fine-tuning
- Medical question-answering with LLMs
- Publishing LoRA adapters on Hugging Face

## Disclaimer

This project is intended for **educational and experimental purposes only**.

It is **not** intended for:

- Clinical diagnosis
- Medical treatment decisions
- Real-world healthcare deployment
- Replacing professional medical judgment

Always consult qualified medical professionals for medical decisions.

## License

Apache License 2.0
