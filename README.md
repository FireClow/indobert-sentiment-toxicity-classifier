# Sentiment and Toxicity Classification of Indonesian Social Media Comments Using IndoBERT

Final project NLP for university coursework.  
This project builds a multitask IndoBERT model to classify:
- Sentiment: `negative`, `neutral`, `positive`
- Toxicity: `non_toxic`, `toxic`

## 1) Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── model_loader.py
│   └── utils.py
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── final_dataset.csv
├── notebooks/
│   └── experimentation.ipynb
├── training/
│   ├── config.py
│   ├── dataset_loader.py
│   ├── prepare_dataset.py
│   ├── train.py
│   └── evaluate.py
├── saved_model/
│   ├── model/
│   └── tokenizer/
├── assets/
│   ├── architecture.png
│   └── confusion_matrix.png
├── requirements.txt
├── .gitignore
└── run.sh
```

## 2) Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt
```

## 3) Dataset Preparation

Place raw files into `dataset/raw/`:
- Indonesian Abusive and Hate Speech Twitter Text
- IndoNLU SmSA
- Optional: IndoToxic2024

Then run:

```bash
python -m training.prepare_dataset
```

Expected output:
- `dataset/final_dataset.csv` with columns:
  - `text`
  - `sentiment_label`
  - `toxicity_label`
  - `source`

## 4) Training

```bash
python -m training.train
```

Expected output:
- Best checkpoint in `saved_model/model/best_model.pt`
- Label map in `saved_model/model/label_mappings.pt`
- Tokenizer files in `saved_model/tokenizer/`

## 5) Evaluation

```bash
python -m training.evaluate
```

Expected output:
- Accuracy, precision, recall, F1-score in terminal
- Confusion matrix figures in `assets/`

## 6) Streamlit Inference App

```bash
streamlit run app/main.py
```

Features:
- text input
- predict button
- sentiment output + confidence
- toxicity output + confidence
- probability bars

## 7) Deployment to Streamlit Cloud

1. Push project to GitHub.
2. In Streamlit Cloud, create app from repository.
3. Set entrypoint: `app/main.py`.
4. Ensure `requirements.txt` is present in repo root.
5. Deploy.

Notes:
- App supports CPU inference by default.
- Model loading is cached using `@st.cache_resource`.

## 8) Architecture Overview

1. Data from multiple datasets is normalized into one multitask dataset.
2. IndoBERT encoder produces shared text representation.
3. Two heads predict sentiment and toxicity simultaneously.
4. Loss is combined and optimized end-to-end.
5. Same trained model is used by Streamlit for inference.

## 9) Presentation Tips

- Show pipeline from raw data -> preprocessing -> multitask model -> UI output.
- Include 5-10 demo comments with varied sentiment/toxicity.
- Explain why multitask learning is efficient (single encoder, two tasks).
- Present both metric tables and confusion matrices.
- Mention practical limitations and future improvements.

