# src/eval.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate

MODEL_PATH = "models/bart-finetuned"   # change if saved elsewhere

# 1. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# 2. Load dataset (CNN/DailyMail test split)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

# 3. Load ROUGE evaluation metric
rouge = evaluate.load("rouge")

preds = []
refs = []

# 4. Evaluate on 200 samples (you can increase later)
for example in dataset.select(range(200)):
    text = example["article"]
    ref = example["highlights"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=128,
        early_stopping=True
    )

    pred = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    preds.append(pred)
    refs.append(ref)

# 5. Compute ROUGE score
result = rouge.compute(
    predictions=preds,
    references=refs
)

print("\n=== ROUGE Evaluation Results ===")
for key, value in result.items():
    print(f"{key}: {value}")
