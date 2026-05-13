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

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

h1 {
    text-align: center;
    color: #00c6ff;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #00c6ff;
    }
    to {
        text-shadow: 0 0 20px #0072ff;
    }
}

.stTextArea textarea {
    border-radius: 15px;
    background-color: #1e1e1e;
    color: white;
}

.stButton button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 250px;
    font-size: 18px;
    border: none;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🎭 AI Sentiment Analysis System")

st.write("Analyze reviews using AI + Machine Learning")
col1, col2, col3 = st.columns(3)

col1.metric("Model", "Logistic Regression")
col2.metric("Dataset", "IMDB 50K")
col3.metric("Accuracy", "89%")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Features")
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

                bg_color = "#14532d"

                st.markdown(f"""
                <div class="result-box"
                     style="background-color:{bg_color};">
                    <h2>😊 Positive Sentiment</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            # Negative Result
            else:

                bg_color = "#7f1d1d"

                st.markdown(f"""
                <div class="result-box"
                     style="background-color:{bg_color};">
                    <h2>😠 Negative Sentiment</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
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