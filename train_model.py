import pandas as pd
import pickle
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")
df=df.sample(25000, random_state=42)
# Text cleaning
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

df['review'] = df['review'].apply(clean_text)
print(df['review'].head(10))
# Features and labels
X = df['review']
y = df['sentiment']

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Trained Successfully!")
