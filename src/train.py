# src/train.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate
import nltk
nltk.download('punkt')

MODEL_NAME = "facebook/bart-base"
OUTPUT_DIR = "./models/bart-finetuned"

# 1. Load dataset (CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 2. Preprocessing function
def preprocess(examples):
    inputs = examples["article"]
    targets = examples["highlights"]

    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=128, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 3. Tokenize dataset
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 4. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    predict_with_generate=True,
    logging_steps=200,
)

# 5. Evaluation metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )
    return rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

# 6. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(2000)),  # small subset to speed up
    eval_dataset=tokenized_dataset["validation"].select(range(500)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. Train
trainer.train()

# 8. Save model
trainer.save_model(OUTPUT_DIR)

