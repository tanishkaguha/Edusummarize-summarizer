# Edusummarize-summarizer
EDUSUMMARIZER is an AI-driven tool that condenses lengthy lecture transcripts or notes into clear, concise summaries. Designed to support students and educators, it highlights key points, reduces redundancy, and makes revision faster and more effective.



# 📚 Edusummarize — AI Text Summarization System

Edusummarize is a deep-learning based **abstractive text summarizer** that uses
Transformer models (BART / T5) to generate concise summaries from long input documents.
It is built using **HuggingFace Transformers**, trained on the **CNN/DailyMail dataset**, 
and includes:

- Model training (`train.py`)
- Inference via command line (`infer.py`)
- Evaluation using ROUGE (`eval.py`)
- Web API using FastAPI (`app.py`)
- Reproducible environment (`requirements.txt` + Dockerfile)

---

# 🚀 Project Motivation

When handling large amounts of study material, research articles, or news coverage,
manually summarizing content is time-consuming.  
Edusummarize aims to **automatically generate high-quality summaries** to save time
for students, educators, and professionals.

---

# 🧠 Features

- Abstractive summarization using BART
- Fine-tuning on CNN/DailyMail
- ROUGE evaluation
- REST API with FastAPI
- Docker support
- Fully reproducible

---

# 🏗 Repository Structure

Edusummarize-Summarizer/
│
├── src/
│ ├── train.py # Fine-tuning BART/T5
│ ├── infer.py # Summarize text from terminal
│ ├── eval.py # Compute ROUGE scores
│ └── app.py # FastAPI web service
│
├── requirements.txt
├── Dockerfile
└── README.md

yaml
Copy code

---

# 📦 Installation

## 1️⃣ Clone the repository
```bash
git clone <your-repo-link>
cd Edusummarize-Summarizer
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
🏋️ Model Training
Run:

bash
Copy code
python src/train.py
The script will:

Load CNN/DailyMail

Tokenize data

Fine-tune BART

Save the model at:

bash
Copy code
models/bart-finetuned/
🔍 Evaluation (ROUGE)
Run:

bash
Copy code
python src/eval.py
Output includes standard metrics:

ROUGE-1

ROUGE-2

ROUGE-L

These scores are reported in the final project report.

✏️ Inference (Generate Summaries)
To summarize custom text:

bash
Copy code
echo "Your text here" | python src/infer.py
or simply run:

bash
Copy code
python src/infer.py
and paste the input text.

🌐 FastAPI Web Demo
Start the API:

bash
Copy code
uvicorn src.app:app --reload --port 8000
Open in browser:

bash
Copy code
http://localhost:8000/docs
Paste the text and click Execute to see the summary.

🐳 Docker (Optional)
Build:

bash
Copy code
docker build -t edusummarize .
Run:

bash
Copy code
docker run -p 8000:8000 edusummarize
📊 Dataset
This project uses:

CNN/DailyMail v3.0.0

Downloaded automatically via:

python
Copy code
from datasets import load_dataset
ds = load_dataset("cnn_dailymail", "3.0.0")
🧪 Evaluation & Analysis
The project reports:

Quantitative metrics (ROUGE)

Errors and sample case studies

Comparison with baselines such as:

Lead-3 extractive baseline

🧩 Future Work
Support multi-document summarization

Parameter-efficient tuning (LoRA / PEFT)

Improve factual consistency

© Acknowledgment
Built using:

HuggingFace Transformers

CNN/DailyMail dataset

PyTorch

This project is created as part of the NLP Course Final Project.
