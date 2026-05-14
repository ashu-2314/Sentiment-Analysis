from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import tempfile
import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import plotly.express as px


from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎭",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TEXT CLEANING ----------------
def clean_text(text):

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = text.lower()

    words = text.split()

    stop_words = set(stopwords.words('english'))

    # Keep negation words
    stop_words.discard("not")
    stop_words.discard("no")
    stop_words.discard("never")

    words = [word for word in words if word not in stop_words]

    # Negation handling
    processed_words = []

    i = 0

    while i < len(words):

        if words[i] in ["not", "no", "never"] and i + 1 < len(words):

            processed_words.append(words[i] + "_" + words[i + 1])

            i += 2

        else:

            processed_words.append(words[i])

            i += 1

    return " ".join(processed_words)

st.markdown("""
<style>

/* ---------------- MAIN APP ---------------- */

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a5f,
        #0f172a
    );
    color: white;
    font-family: 'Segoe UI', sans-serif;
    padding-bottom: 80px;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ---------------- TITLE ---------------- */

h1 {
    text-align: center;
    color: #ffffff;
    font-size: 58px !important;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 10px;

    text-shadow:
        0 0 6px rgba(0,198,255,0.45),
        0 0 12px rgba(0,114,255,0.25);

    animation: titleGlow 3s ease-in-out infinite alternate;
}
@keyframes titleGlow {

    from {
        text-shadow:
            0 0 6px rgba(0,198,255,0.35),
            0 0 12px rgba(0,114,255,0.20);
    }

    to {
        text-shadow:
            0 0 10px rgba(0,198,255,0.55),
            0 0 18px rgba(0,114,255,0.35);
    }
}
/* ---------------- SUBTEXT ---------------- */

.subtext {
    text-align: center;
    font-size: 20px;
    color: #d1d5db;
    margin-bottom: 40px;
}

/* ---------------- GLASS CARD ---------------- */

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 25px;
    margin-top: 25px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* ---------------- TEXT AREA ---------------- */

textarea {
    background-color: rgba(0,0,0,0.45) !important;
    color: white !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    font-size: 18px !important;
}

/* ---------------- BUTTON ---------------- */

.stButton > button {
    width: 100%;
    border-radius: 16px;
    height: 60px;
    border: none;
    font-size: 20px;
    font-weight: bold;
    color: white;
    background: linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );
    transition: 0.3s ease;
    box-shadow: 0 6px 18px rgba(0,114,255,0.4);
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 8px 25px rgba(0,198,255,0.7);
}

/* ---------------- RESULT BOX ---------------- */

.result-box {

    padding: 35px;

    border-radius: 28px;

    margin-top: 30px;

    color: white;

    text-align: center;

    animation: fadeIn 0.7s ease-in-out;

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow: 0 10px 35px rgba(0,0,0,0.35);

    transition: 0.4s ease;
}

.result-box:hover {

    transform: translateY(-4px);

    box-shadow: 0 14px 45px rgba(0,0,0,0.45);
}

.result-box h1 {

    text-shadow:
        0 0 12px rgba(255,255,255,0.25);

    letter-spacing: 1px;
}

/* ---------------- ANIMATION ---------------- */

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* ---------------- METRICS ---------------- */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ---------------- FOOTER ---------------- */

.footer {

    position: fixed;

    bottom: 0;

    left: 0;

    width: 100%;

    background: rgba(15, 23, 42, 0.95);

    backdrop-filter: blur(10px);

    border-top: 1px solid rgba(255,255,255,0.08);

    text-align: center;

    padding: 12px;

    color: #d1d5db;

    font-size: 15px;

    z-index: 999;

    box-shadow: 0 -4px 15px rgba(0,0,0,0.25);
}

/* ---------------- SCROLLBAR ---------------- */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #00c6ff;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1>🎭 AI Sentiment Analysis System</h1>
<p class="subtext">
Analyze reviews using AI + Machine Learning
</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("Model", "Logistic Regression")
col2.metric("Dataset", "IMDB 50K")
col3.metric("Accuracy", "89%")


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("""
<h2 style='
color:white;
text-align:center;
margin-top:10px;
margin-bottom:20px;
'>
⚙ Features
</h2>
""", unsafe_allow_html=True)

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
    width=80
)
voice_option = st.sidebar.checkbox("Enable Voice Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------- VOICE INPUT ----------------

review = ""

if voice_option:

    st.subheader("🎤 Smart Voice Input")

    st.info("Click microphone to start and stop recording.")

    audio_bytes = audio_recorder(
        text="",
        recording_color="#ff4b4b",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x"
    )

    if audio_bytes:

        st.success("✅ Recording completed!")

        recognizer = sr.Recognizer()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:

            f.write(audio_bytes)

            temp_audio_path = f.name

        try:

            with sr.AudioFile(temp_audio_path) as source:

                audio_data = recognizer.record(source)

                review = recognizer.recognize_google(audio_data)

                st.subheader("📝 Converted Speech To Text")

                st.text_area(
                    "Recognized Text",
                    value=review,
                    height=150
                )

        except Exception as e:

            st.error(f"Speech recognition failed: {e}")

# ---------------- TEXT AREA ----------------
review = st.text_area(
    "Enter Your Review",
    value=review if review else "",
    height=150
)

# ---------------- PREDICT ----------------
if st.button("🚀 Analyze Sentiment"):

    if review.strip() != "":

        with st.spinner("Analyzing Sentiment..."):

            cleaned_review = clean_text(review)

            st.write("Cleaned Text:", cleaned_review)

            vector_input = vectorizer.transform([cleaned_review])

            prediction = model.predict(vector_input)[0]

            probability = model.predict_proba(vector_input)

            confidence = max(probability[0]) * 100

            # Save History
            st.session_state.history.append(prediction)

            # Positive Result
            if prediction == "positive":
                bg_color = " linear-gradient(135deg,rgba(22,163,74,0.95),rgba(21,128,61,0.88))"

                st.markdown(f"""
                <div class="result-box"
                style="background:{bg_color}">

                    <h1 style='
                    font-size:52px;
                    margin-bottom:15px;
                    text-shadow:0 0 18px rgba(255,255,255,0.3);
                    '>
                    😊 Positive Sentiment
                    </h1>

                    <h2 style='
                    font-size:34px;
                    margin-top:15px;
                    '>
                    Confidence: {confidence:.2f}%
                    </h2>

                </div>
                """, unsafe_allow_html=True)

            # Negative Result
            else:

                bg_color = "linear-gradient(135deg,rgba(220,38,38,0.95),rgba(127,29,29,0.88))"

                st.markdown(f"""
                <div class="result-box"
                style="background:{bg_color}">

                    <h1 style='
                    font-size:52px;
                    margin-bottom:15px;
                    text-shadow:0 0 18px rgba(255,255,255,0.3);
                    '>
                    😠 Negative Sentiment
                    </h1>

                    <h2 style='
                    font-size:34px;
                    margin-top:15px;
                    '>
                    Confidence: {confidence:.2f}%
                    </h2>
 
                </div>
                """, unsafe_allow_html=True)

            # Progress Bar
            st.progress(int(confidence))

            # Word Cloud
            st.subheader("☁ Word Cloud")

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='black'
            ).generate(cleaned_review)

            fig, ax = plt.subplots()

            ax.imshow(wordcloud)

            ax.axis("off")

            st.pyplot(fig)

# ---------------- PIE CHART ----------------
if len(st.session_state.history) > 0:

    st.subheader("📊 Sentiment History")

    counts = Counter(st.session_state.history)

    chart_data = pd.DataFrame({
        "Sentiment": list(counts.keys()),
        "Count": list(counts.values())
    })

    fig = px.pie(
        chart_data,
        names='Sentiment',
        values='Count',
        title='Sentiment Distribution'
    )

    st.plotly_chart(fig)

# ---------------- CSV UPLOAD ----------------
if uploaded_file is not None:

    st.subheader("📁 CSV Analysis")

    df = pd.read_csv(uploaded_file)

    st.write(df.head())

    if "review" in df.columns:

        predictions = []

        for text in df["review"]:

            cleaned = clean_text(str(text))

            vector = vectorizer.transform([cleaned])

            pred = model.predict(vector)[0]

            predictions.append(pred)

        df["Prediction"] = predictions

        st.write(df)

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇ Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )
        st.markdown("""
<hr>
<center>
Built with ❤️ using Python, NLP & Machine Learning
</center>
""", unsafe_allow_html=True)
        
st.markdown("""
<div class="footer">
    🚀 Built with Python • NLP • Streamlit • Machine Learning
</div>
""", unsafe_allow_html=True)     