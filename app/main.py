import streamlit as st

from app.inference import predict_text
from app.model_loader import load_model_and_tokenizer

st.set_page_config(page_title="IndoBERT Sentiment + Toxicity", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner=True)
def get_pipeline():
    model, tokenizer, device, label_maps = load_model_and_tokenizer()
    sentiment_id2label = {int(k): v for k, v in label_maps["sentiment_id2label"].items()}
    toxicity_id2label = {int(k): v for k, v in label_maps["toxicity_id2label"].items()}
    return model, tokenizer, device, sentiment_id2label, toxicity_id2label


def render_result_card(title: str, label: str, confidence: float):
    st.markdown(
        f"""
        <div style='padding:1rem;border-radius:12px;border:1px solid #3b82f6;margin-bottom:0.8rem;'>
            <h4 style='margin:0;'>{title}</h4>
            <p style='font-size:1.1rem;margin:0.4rem 0 0.2rem 0;'><b>{label}</b></p>
            <p style='margin:0;color:#9ca3af;'>Confidence: {confidence:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("Sentiment and Toxicity Classification (IndoBERT)")
    st.caption("Final Project NLP - Indonesian Social Media Comments")

    model, tokenizer, device, sentiment_id2label, toxicity_id2label = get_pipeline()
    st.success(f"Model loaded on: {device.upper()}")

    text = st.text_area(
        "Masukkan komentar:",
        height=160,
        placeholder="Contoh: Film ini bagus banget, recommended!",
    )

    predict_btn = st.button("Predict", type="primary")
    if predict_btn:
        if not text.strip():
            st.warning("Input teks tidak boleh kosong.")
            return

        result = predict_text(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            sentiment_id2label=sentiment_id2label,
            toxicity_id2label=toxicity_id2label,
        )

        st.subheader("Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            render_result_card("Sentiment", result.sentiment_label, result.sentiment_confidence)
        with col2:
            render_result_card("Toxicity", result.toxicity_label, result.toxicity_confidence)

        st.write("Preprocessed text:")
        st.code(result.cleaned_text)

        st.subheader("Probabilities")
        st.write("Sentiment")
        st.bar_chart(result.sentiment_probabilities)
        st.write("Toxicity")
        st.bar_chart(result.toxicity_probabilities)


if __name__ == "__main__":
    main()

