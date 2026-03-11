import pandas as pd
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from collections import Counter
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print(" Project started")

# 1. LOAD DATA (LIMIT TO 5000)

try:
    df = pd.read_csv("tweets.csv")
    df = df.head(5000)
    df.rename(columns={"Tweet": "raw_text"}, inplace=True)
    print(f" Dataset size used: {len(df)}")
except FileNotFoundError:
    print(" Error: 'tweets.csv' not found. Please upload the file.")
    exit()

# 2. CLEAN TEXT

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    # Keeping English and Hindi characters (Devanagari range)
    text = re.sub(r"[^a-zA-Zअ-ह\s]", "", text)
    return text

df["clean_text"] = df["raw_text"].apply(clean_text)

# 3. LANGUAGE DETECTION

def detect_lang(text):
    try:
        # Only detect if text is long enough to avoid errors
        if len(text) > 3:
            return detect(text)
        return "unknown"
    except:
        return "unknown"

print(" Detecting languages...")
df["language"] = df["clean_text"].apply(detect_lang)

# 4. VADER SENTIMENT

vader = SentimentIntensityAnalyzer()

def vader_score(text, lang):
    # VADER works best on English
    if lang == "en":
        return vader.polarity_scores(text)["compound"]
    return np.nan

df["vader_sentiment"] = df.apply(
    lambda x: vader_score(x["clean_text"], x["language"]), axis=1
)

# 5. BERT SENTIMENT (CPU)

print(" Loading BERT model (CPU)... this may take a moment.")

bert = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    # Use device=0 if you have a GPU, otherwise -1 for CPU
    device=-1
)

def bert_sentiment(text):
    try:
        # Truncate to 512 tokens to fit BERT limits
        result = bert(text[:512])[0]
        stars = int(result["label"][0])
        # Convert 1-5 stars to -1 to 1 scale
        return (stars - 3) / 2
    except:
        return 0

# Apply BERT to all rows (filling NaNs from VADER or refining them)
df["bert_sentiment"] = df["clean_text"].apply(bert_sentiment)

# 6. FINAL SENTIMENT

df["final_sentiment"] = df[["vader_sentiment", "bert_sentiment"]].mean(axis=1)
df["final_sentiment"] = df["final_sentiment"].fillna(df["bert_sentiment"])

# 7. ASSIGN PARTIES (SIMULATION)

parties = ["Party A", "Party B"]
df["party"] = [random.choice(parties) for _ in range(len(df))]

# 8. PREPARE TRAINING DATA (SIMULATION)

df["simulated_actual_vote"] = (df["final_sentiment"] * 20) + 50


# 9. REGRESSION MODEL

X = df[["final_sentiment"]]
y = df["simulated_actual_vote"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(f" Model Coefficient: {model.coef_[0]:.2f}")
print(f" Model Intercept: {model.intercept_:.2f}")

# Predict on the whole dataset
df["predicted_vote_share"] = model.predict(X)

# 10. SAVE MAIN OUTPUT

df.to_csv("final_prediction_output.csv", index=False)
print("Prediction output saved to 'final_prediction_output.csv'")

# 11. VISUALIZATION

vader_vals = df["vader_sentiment"].dropna()
bert_vals = df["bert_sentiment"].dropna()

bins = np.linspace(-1, 1, 6)
labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

vader_counts, _ = np.histogram(vader_vals, bins=bins)
bert_counts, _ = np.histogram(bert_vals, bins=bins)

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, vader_counts, width, label="VADER (Rule-based)", color='skyblue')
plt.bar(x + width/2, bert_counts, width, label="BERT (Deep Learning)", color='orange')

plt.xlabel("Sentiment Category")
plt.ylabel("Number of Tweets")
plt.title("Sentiment Distribution: VADER vs BERT")
plt.xticks(x, labels)
plt.legend()

plt.savefig("sentiment_analysis_plot.png")
plt.close()
print("Sentiment graph saved as 'sentiment_analysis_plot.png'")

# 12. FINAL ELECTION PREDICTION RESULTS

print("\n" + "="*40)
print(" FINAL ELECTION FORECAST")
print("="*40)

# Group by Party and take the average of the predicted vote shares
election_result = df.groupby("party")["predicted_vote_share"].mean().reset_index()

# Sort by highest vote share
election_result = election_result.sort_values(by="predicted_vote_share", ascending=False)

for index, row in election_result.iterrows():
    print(f" {row['party']}: {row['predicted_vote_share']:.2f}% Vote Share")

winner = election_result.iloc[0]['party']
print(f"\n PROJECTED WINNER: {winner}")
print("="*40)

# 13. KEYWORD ANALYSIS

words = []
stopwords = ["this", "that", "with", "from", "have", "what"] # Add more as needed

for text in df["clean_text"]:
    # meaningful words > 4 chars
    found_words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    words.extend([w for w in found_words if w not in stopwords])

top_keywords = Counter(words).most_common(20)
keyword_df = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])
keyword_df.to_csv("top_keywords.csv", index=False)

print("Keyword analysis saved as 'top_keywords.csv'")
print("PROJECT COMPLETED SUCCESSFULLY")
