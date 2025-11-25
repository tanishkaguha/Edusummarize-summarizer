# src/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/bart-finetuned"  # Change if needed

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# FastAPI app
app = FastAPI(
    title="Edusummarize — Text Summarization API",
    description="Submit text and receive a generated summary."
)

# Request format
class InputText(BaseModel):
    text: str

# POST /summarize endpoint
@app.post("/summarize")
async def summarize(input_data: InputText):
    inputs = tokenizer(
        input_data.text,
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

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return {
        "summary": summary
    }
