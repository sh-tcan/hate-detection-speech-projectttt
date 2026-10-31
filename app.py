import streamlit as st
import joblib
import re
from pathlib import Path

st.set_page_config(
    page_title="ClearSpace AI | Hate Speech Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path("hate_speech_model.pkl")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; color: #182230; }
.block-container { max-width: 1120px; padding-top: 1.4rem; padding-bottom: 3rem; }
.hero {
  background: linear-gradient(120deg, #101c3a 0%, #173b66 55%, #176b70 100%);
  padding: 42px 44px; border-radius: 24px; color: white; margin-bottom: 22px;
}
.hero h1 { font-family: Manrope,sans-serif; font-size: 42px; line-height: 1.1; margin: 8px 0 12px; color: white; }
.hero p { color: #dbeafe; font-size: 16px; max-width: 720px; }
.eyebrow { color: #9ce8dc; font-weight: 700; letter-spacing: 2px; font-size: 12px; }
.panel {
  background: white; padding: 24px; border: 1px solid #e5eaf1;
  border-radius: 18px; box-shadow: 0 8px 24px rgba(20,40,70,.04);
}
.small-label { color: #607086; text-transform: uppercase; letter-spacing: 1.3px; font-size: 11px; font-weight: 700; }
.metric { font-family: Manrope,sans-serif; font-size: 27px; font-weight: 800; color: #172b4d; }
.muted { color: #64748b; }
.result {
  padding: 18px 20px; border-radius: 14px; margin: 12px 0;
  border: 1px solid #dce5ee; background: #f8fafc;
}
.result-title { font-size: 21px; font-weight: 800; margin-bottom: 5px; }
div.stButton > button {
  background: #176b70; color: white; border: 0; border-radius: 10px;
  padding: 10px 22px; font-weight: 700; min-height: 46px;
}
div.stButton > button:hover { background: #12575b; color: white; border: 0; }
div[data-testid="stTextArea"] textarea { border-radius: 12px; border: 1px solid #cbd5e1; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

st.markdown("""
<div class="hero">
  <div class="eyebrow">THINK BEFORE YOU POST</div>
  <h1>ClearSpace AI</h1>
  <p>A student-built Natural Language Processing project that analyses online text and helps identify potentially harmful language. Promoting safer, more respectful digital spaces through AI.</p>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns([1,1,1])
with nav1:
    st.markdown('<div class="panel"><div class="small-label">AI DOMAIN</div><div class="metric">NLP</div><div class="muted">Understanding written language</div></div>', unsafe_allow_html=True)
with nav2:
    st.markdown('<div class="panel"><div class="small-label">ML APPROACH</div><div class="metric">Supervised</div><div class="muted">Learning from labelled examples</div></div>', unsafe_allow_html=True)
with nav3:
    st.markdown('<div class="panel"><div class="small-label">CLASSIFIER</div><div class="metric">Decision Tree</div><div class="muted">Text classification algorithm</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.45, 0.8], gap="large")

with left:
    st.markdown("### Analyse a message")
    st.markdown('<p class="muted">Paste a social media comment, tweet, or short message below. The model will predict its category.</p>', unsafe_allow_html=True)
    user_text = st.text_area("Text to analyse", height=170, placeholder="Type or paste a message here...", label_visibility="collapsed")
    analyse = st.button("🛡️  Analyse Text", use_container_width=True)

    if analyse:
        if not user_text.strip():
            st.warning("Please enter some text before analysing it.")
        elif model is None:
            st.error("The trained AI model has not been added yet. Follow the setup instructions in the README to train the model using twitter_data.csv.")
        else:
            prediction = int(model.predict([clean_text(user_text)])[0])
            label_map = {
                0: ("Hate speech detected", "The model found patterns associated with hate speech.", "error"),
                1: ("Offensive language detected", "The model found patterns associated with offensive language.", "warning"),
                2: ("Neither detected", "The model classified this text as neither hate speech nor offensive language.", "success")
            }
            title, description, kind = label_map.get(prediction, ("Unclassified result", "The model returned an unexpected class.", "info"))
            if kind == "error":
                st.error("**" + title + "**\n\n" + description)
            elif kind == "warning":
                st.warning("**" + title + "**\n\n" + description)
            elif kind == "success":
                st.success("**" + title + "**\n\n" + description)
            else:
                st.info("**" + title + "**\n\n" + description)
            st.caption("This is an automated prediction, not a final judgement. Review the context before taking action.")

with right:
    st.markdown("### How it works")
    st.markdown("""
    <div class="panel">
      <div class="small-label">THE PROCESS</div>
      <p><b>01 · Input</b><br><span class="muted">A user enters a message.</span></p>
      <p><b>02 · Pre-processing</b><br><span class="muted">Text is cleaned and converted into numerical features.</span></p>
      <p><b>03 · Prediction</b><br><span class="muted">The trained Decision Tree model analyses the features.</span></p>
      <p><b>04 · Result</b><br><span class="muted">The predicted category is displayed.</span></p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("### The three classification categories")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="panel"><div class="small-label">CLASS 0</div><h3>Hate speech</h3><p class="muted">Language that attacks, threatens, or promotes prejudice against people based on identity or group membership.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="panel"><div class="small-label">CLASS 1</div><h3>Offensive language</h3><p class="muted">Insulting, abusive, or vulgar language that may not necessarily target a protected group.</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="panel"><div class="small-label">CLASS 2</div><h3>Neither</h3><p class="muted">Text that the model does not classify as hate speech or offensive language.</p></div>', unsafe_allow_html=True)

st.write("")
a, b = st.columns([1,1], gap="large")
with a:
    st.markdown("""
    <div class="panel">
      <h3>Why does this matter?</h3>
      <p class="muted">Online hate and abusive language can make digital spaces unsafe and discourage people from participating. ClearSpace AI is designed as an educational prototype to raise awareness and support careful review of online content.</p>
    </div>
    """, unsafe_allow_html=True)
with b:
    st.markdown("""
    <div class="panel">
      <h3>Responsible AI notice</h3>
      <p class="muted">Language is complex. Context, sarcasm, dialects, and reclaimed words can affect predictions. This model can make mistakes and may reflect bias in its training data. It should not automatically punish, report, or silence anyone.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#718096;font-size:13px'>CLEARSPACE AI · CBSE CLASS XII ARTIFICIAL INTELLIGENCE PROJECT · SUBJECT CODE 843</div>", unsafe_allow_html=True)
