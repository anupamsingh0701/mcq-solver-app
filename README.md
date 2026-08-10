# AI Multiple-Choice Answering System

An intelligent Multiple-Choice Question (MCQ) Answering Web Application powered by **DeBERTa-v3-Large**, **RoBERTa-Large (RACE)**, and custom fine-tuned transformer models built with **Streamlit** and **PyTorch**.

##  Features
- **Interactive Single Question Solver**: Enter any question with 5 options (A-E) to get predictions, confidence metrics, and MAP@3 rankings.
- **Batch CSV Inference**: Upload a CSV file containing multiple questions to generate bulk predictions.
- **Private HuggingFace Repositories**: Native support for downloading and running private fine-tuned model checkpoints using HF Access Tokens.

##  Local Setup

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
