import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()
sia = SentimentIntensityAnalyzer()

FEATURE_COLS = [
    'Retweet Count', 'Mention Count', 'Follower Count', 'Verified',
    'has_hashtag', 'hour',
    'tweet_length', 'url_presence', 'lexical_diversity', 'hashtag_count',
    'sentiment', 'account_age_days',
]

st.title("Bot or Not?")
st.markdown("Enter a Twitter account's details to get a bot probability score.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        follower_count = st.number_input("Follower Count", min_value=0, value=1000)
        retweet_count = st.number_input("Retweet Count", min_value=0, value=10)
        mention_count = st.number_input("Mention Count", min_value=0, value=2)
        verified = st.selectbox("Verified", ["No", "Yes"])
        hour = st.slider("Posting Hour (0–23)", min_value=0, max_value=23, value=12)

    with col2:
        tweet_text = st.text_area("Tweet Text", value="Excited to share this new article about technology!")
        hashtags = st.text_input("Hashtags (space-separated)", value="tech news")
        created_year = st.number_input("Account Created — Year", min_value=2006, max_value=2026, value=2020)
        created_month = st.selectbox("Account Created — Month", list(range(1, 13)), index=0, format_func=lambda m: pd.Timestamp(2000, m, 1).strftime('%B'))

    submitted = st.form_submit_button("Check Account")

if submitted:
    # Compute features
    tweet_length = len(tweet_text)
    url_presence = int('http' in tweet_text.lower())
    words = tweet_text.lower().split()
    lexical_diversity = len(set(words)) / max(len(words), 1)
    hashtag_count = len(hashtags.split()) if hashtags.strip() else 0
    has_hashtag = int(hashtag_count > 0)
    sentiment = sia.polarity_scores(tweet_text)['compound']
    account_age_days = (pd.Timestamp.today() - pd.Timestamp(year=created_year, month=created_month, day=1)).days
    verified_int = 1 if verified == "Yes" else 0

    features = pd.DataFrame([[
        retweet_count, mention_count, follower_count, verified_int,
        has_hashtag, hour,
        tweet_length, url_presence, lexical_diversity, hashtag_count,
        sentiment, account_age_days,
    ]], columns=FEATURE_COLS)

    prob = model.predict_proba(features)[0][1]

    st.divider()

    if prob >= 0.5:
        st.error(f"Bot probability: **{prob:.0%}**  — likely a bot")
    else:
        st.success(f"Bot probability: **{prob:.0%}**  — likely human")

    st.progress(float(prob))

    # SHAP waterfall for this prediction
    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=features.iloc[0].values,
            feature_names=FEATURE_COLS,
        ),
        show=False,
    )
    plt.tight_layout()
    st.pyplot(fig)
