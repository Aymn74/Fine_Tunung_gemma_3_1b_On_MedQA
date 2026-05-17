# -*- coding: utf-8 -*-
"""
Fine-tuning Gemma 3 1B on MedQA using Unsloth and LoRA.

This script demonstrates:
- Loading a medical QA dataset from Hugging Face
- Formatting examples into instruction-style prompts
- Fine-tuning Gemma 3 1B with LoRA using Unsloth
- Testing inference
- Saving the LoRA adapter locally

Hugging Face model page:
https://huggingface.co/Ay74/medqa-gemma-1b-lora
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


BASE_MODEL = "unsloth/gemma-3-1b-it"
MAX_SEQ_LENGTH = 1024
TRAINING_SAMPLES = 500
OUTPUT_DIR = "outputs"
LOCAL_ADAPTER_DIR = "medqa_gemma_lora"


def check_gpu() -> None:
    """Check GPU availability."""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def load_base_model():
    """Load the base Gemma model using Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    print("Model loaded successfully.")
    return model, tokenizer


def format_medqa(example):
    """Convert MedQA rows into Gemma chat-style training text."""
    options_text = "\n".join(
        [f"{key}. {value}" for key, value in example["options"].items()]
    )

    prompt = f"""Answer the following medical multiple-choice question.

Question:
{example["question"]}

Options:
{options_text}

Return the correct option letter and answer only.
"""

    response = f"{example['answer_idx']}. {example['answer']}"

    return {
        "text": (
            f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}<end_of_turn>"
        )
    }


def prepare_dataset():
    """Load and format the MedQA dataset."""
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    train_dataset = dataset["train"].select(range(TRAINING_SAMPLES)).map(format_medqa)
    print(train_dataset[0]["text"])
    return train_dataset


def apply_lora(model):
    """Apply LoRA adapters to the model."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model


def train_model(model, tokenizer, train_dataset):
    """Fine-tune the model using SFTTrainer."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            seed=3407,
        ),
    )

    trainer.train()
    return model


def test_inference(model, tokenizer):
    """Run a simple medical QA inference test."""
    FastLanguageModel.for_inference(model)

    prompt = """<bos><start_of_turn>user
Answer the following medical multiple-choice question.

Question:
A pregnant woman has dysuria without fever or flank pain. What is the best treatment?

Options:
A. Doxycycline
B. Nitrofurantoin
C. Vancomycin
D. Oseltamivir

Return the correct option letter and answer only.
<end_of_turn>
<start_of_turn>model
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.1,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


def save_adapter(model, tokenizer):
    """Save the LoRA adapter locally."""
    model.save_pretrained(LOCAL_ADAPTER_DIR)
    tokenizer.save_pretrained(LOCAL_ADAPTER_DIR)
    print(f"LoRA adapter saved to: {LOCAL_ADAPTER_DIR}")


def main():
    check_gpu()
    model, tokenizer = load_base_model()
    train_dataset = prepare_dataset()
    model = apply_lora(model)
    model = train_model(model, tokenizer, train_dataset)
    test_inference(model, tokenizer)
    save_adapter(model, tokenizer)


if __name__ == "__main__":
    main()
