import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from inference import predict_text
from model_loader import load_model_and_tokenizer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IndoBERT Sentiment & Toxicity", page_icon="✨", layout="centered")

# --- ADVANCED CSS INJECTION (NUTRICHECK THEME + ANIMATION) ---
st.markdown("""
<style>
    /* Keyframe Animations */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Background Gradient (NutriCheck Vibe) */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0b0f19 0%, #171026 90%);
    }

    /* Kustomisasi Input Area */
    .stTextArea textarea {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #7e22ce !important;
        box-shadow: 0 0 0 1px #7e22ce !important;
    }

    /* Kustomisasi Tombol Predict (Gradient Purple) */
    .stButton > button {
        background: linear-gradient(90deg, #4c1d95 0%, #7e22ce 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(126, 34, 206, 0.4) !important;
    }

    /* Glassmorphism Card (Dark Mode + Animation) */
    .glass-card {
        background: rgba(22, 27, 34, 0.7); 
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideUpFade 0.6s ease-out forwards;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(126, 34, 206, 0.2);
    }

    /* Sentiment & Toxicity Specific Accents */
    .border-positive { border-left: 4px solid #2ea043; }
    .border-negative { border-left: 4px solid #da3633; }
    .border-neutral { border-left: 4px solid #d29922; }

    /* Typography inside cards */
    .card-title { font-size: 1rem; font-weight: 500; opacity: 0.7; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;}
    .card-value { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }
    .card-confidence { font-size: 0.85rem; opacity: 0.6; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def get_pipeline():
    model, tokenizer, device, label_maps = load_model_and_tokenizer()
    sentiment_id2label = {int(k): v for k, v in label_maps["sentiment_id2label"].items()}
    toxicity_id2label = {int(k): v for k, v in label_maps["toxicity_id2label"].items()}
    return model, tokenizer, device, sentiment_id2label, toxicity_id2label


def main():
    # --- HEADER ---
    st.title("✨ Social Media Text Analyzer")
    st.caption("Powered by IndoBERT • Natural Language Processing Final Project")

    model, tokenizer, device, sentiment_id2label, toxicity_id2label = get_pipeline()
    
    with st.expander("⚙️ System Diagnostics", expanded=False):
        st.success(f"Model Engine Active. Running on: {device.upper()}")

    # --- INPUT ZONE ---
    text = st.text_area(
        label="Analyze Statement",
        label_visibility="hidden",
        height=140,
        placeholder="Ketik komentar media sosial di sini untuk dianalisis...",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🔍 Run NLP Engine", type="primary", use_container_width=True)
    
    if predict_btn:
        if not text.strip():
            st.warning("Mohon masukkan teks terlebih dahulu.")
            return

        with st.spinner("Processing semantics and toxicity metrics..."):
            result = predict_text(text, model, tokenizer, device, sentiment_id2label, toxicity_id2label)

        st.markdown("<br><h3 style='text-align: center;'>Analysis Result</h3><br>", unsafe_allow_html=True)
        
        # --- LOGIC STYLING ---
        # Sentiment
        if result.sentiment_label.lower() == "positive":
            sent_class = "border-positive"
            sent_color = "#2ea043"
            sent_icon = "✨"
        elif result.sentiment_label.lower() == "negative":
            sent_class = "border-negative"
            sent_color = "#da3633"
            sent_icon = "⚠️"
        else:
            sent_class = "border-neutral"
            sent_color = "#d29922"
            sent_icon = "⚖️"

        # Toxicity
        if result.toxicity_label.lower() == "toxic":
            tox_class = "border-negative"
            tox_color = "#da3633"
            tox_icon = "☣️"
        else:
            tox_class = "border-positive"
            tox_color = "#2ea043"
            tox_icon = "🛡️"

        # --- RENDER GLASS CARDS ---
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown(f"""
            <div class="glass-card {sent_class}">
                <div class="card-title">{sent_icon} Sentiment Context</div>
                <div class="card-value" style="color: {sent_color};">{result.sentiment_label}</div>
                <div class="card-confidence">Confidence Level: <b>{result.sentiment_confidence:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown(f"""
            <div class="glass-card {tox_class}">
                <div class="card-title">{tox_icon} Toxicity Detection</div>
                <div class="card-value" style="color: {tox_color};">{result.toxicity_label}</div>
                <div class="card-confidence">Confidence Level: <b>{result.toxicity_confidence:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- PLOTLY CHARTS (CUSTOM COLORS & HIGHLIGHT) ---
        st.markdown("#### 📈 Probabilities Breakdown")
        
        df_sent = pd.DataFrame(list(result.sentiment_probabilities.items()), columns=['Class', 'Probability'])
        df_tox = pd.DataFrame(list(result.toxicity_probabilities.items()), columns=['Class', 'Probability'])

        chart_col1, chart_col2 = st.columns(2)

        def create_highlighted_chart(df, color_mapping):
            max_val = df['Probability'].max() # Mencari nilai tertinggi untuk di-highlight
            colors = []
            opacities = []
            
            for _, row in df.iterrows():
                # Normalisasi string agar cocok dengan key di dictionary
                cls_name = str(row['Class']).lower().replace('-', '_')
                colors.append(color_mapping.get(cls_name, "#888888"))
                
                # Highlight bar tertinggi
                if row['Probability'] == max_val:
                    opacities.append(1.0)
                else:
                    opacities.append(0.5) # Ditingkatkan menjadi 0.5 agar warna terang tidak tersedot background gelap

            fig = go.Figure(go.Bar(
                x=df['Probability'],
                y=df['Class'],
                orientation='h',
                marker=dict(
                    color=colors, 
                    opacity=opacities,
                    line=dict(color=colors, width=2) # Border agar bar yang meredup tetap terlihat jelas
                ),
                text=[f"{val:.1%}" for val in df['Probability']],
                textposition='outside',
                textfont=dict(color="#e2e8f0", size=14)
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",  
                font_color="#e2e8f0",          
                margin=dict(l=0, r=45, t=10, b=0), # Margin kanan ditambah agar teks persentase muat
                height=180,
                showlegend=False,
                xaxis=dict(range=[0, 1.15], showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=""),
                yaxis=dict(title="")
            )
            return fig

        with chart_col1:
            sent_colors = {
                'positive': '#3b82f6', # Biru
                'neutral': '#f8fafc',  # Putih
                'negative': '#ef4444'  # Merah
            }
            st.plotly_chart(create_highlighted_chart(df_sent, sent_colors), use_container_width=True)

        with chart_col2:
            tox_colors = {
                'toxic': '#ef4444',      # Merah
                'non_toxic': '#4ade80',  # Hijau Terang (Neon)
                'non toxic': '#4ade80'   # Hijau Terang (Neon)
            }
            st.plotly_chart(create_highlighted_chart(df_tox, tox_colors), use_container_width=True)

        with st.expander("🔍 View Preprocessing Logs"):
            st.code(result.cleaned_text, language="text")

if __name__ == "__main__":
    main()