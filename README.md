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

Model weights (`best_model.pt`, ~475 MB) are stored with **Git LFS** in this repo.

1. Ensure Git LFS is installed locally (`git lfs install`).
2. Push the repo (including LFS objects) to GitHub.
3. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**.
4. Repository: `FireClow/indobert-sentiment-toxicity-classifier`
5. Branch: `main`
6. Main file path: `app/main.py`
7. Deploy (first build may take several minutes while LFS downloads the model).

**Optional — Hugging Face Hub instead of Git LFS**

```bash
set HF_TOKEN=hf_...
python scripts/push_model_to_hub.py --repo-id YOUR_USERNAME/indobert-sentiment-toxicity-multitask
```

In Streamlit Cloud → **Settings → Secrets**:

```toml
HF_MODEL_REPO = "YOUR_USERNAME/indobert-sentiment-toxicity-multitask"
```

**Notes**
- App uses CPU inference on Streamlit Cloud.
- Model loading is cached with `@st.cache_resource`.
- Free tier has ~1 GB RAM; if the app crashes on startup, use a machine with more memory or the HF Hub option.

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

