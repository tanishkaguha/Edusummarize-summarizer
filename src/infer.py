# src/infer.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

MODEL_PATH = "models/bart-finetuned"   # change if needed

# 1. Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# 2. Read input text
if sys.stdin.isatty():
    text = input("Enter text to summarize:\n")
else:
    text = sys.stdin.read()

# 3. Run summarization
inputs = tokenizer(
    text, return_tensors="pt",
    truncation=True, max_length=1024
)

summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=128,
    early_stopping=True
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\n=== SUMMARY ===\n")
print(summary)
