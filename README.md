# MedQA Gemma 1B LoRA Fine-Tuning

This project demonstrates lightweight medical large language model (LLM) fine-tuning using **Gemma 3 1B**, **Unsloth**, and **LoRA** on the **MedQA-USMLE** dataset.

The workflow includes:

- Loading a medical QA dataset from Hugging Face
- Formatting data into instruction-style prompts
- Applying LoRA adapters
- Fine-tuning with Unsloth and TRL
- Testing inference
- Saving the LoRA adapter locally
- Publishing the trained adapter on Hugging Face

## Hugging Face Model

The trained LoRA adapter is available here:

https://huggingface.co/Ay74/medqa-gemma-1b-lora

## Base Model

```text
unsloth/gemma-3-1b-it
```

## Dataset

```text
GBaker/MedQA-USMLE-4-options
```

## Training Summary

| Item | Value |
|---|---|
| Base Model | Gemma 3 1B |
| Fine-tuning Method | LoRA |
| Framework | Unsloth |
| Dataset | MedQA-USMLE 4-options |
| Training Samples | 500 |
| Max Steps | 30 |
| GPU | Tesla T4 |
| Platform | Google Colab |

## Installation

```bash
pip install unsloth datasets transformers trl peft accelerate bitsandbytes
```

## Usage

Run the training script:

```bash
python fine_tuning_gemma_1b_medqa.py
```

## Example Output

```text
B. Nitrofurantoin
```

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

## License

Apache License 2.0
