# Bot vs Human Twitter Classifier — Project Context

## What this project is
A machine learning classifier that distinguishes bot accounts from real humans on Twitter/X using behavioral patterns, account metadata, and tweet language features.

## Dataset
- Source: Kaggle (pre-labeled bot/human accounts)
- Format: Single CSV file with account metadata + labels
- Target column: `label` (bot / human)

## Tech stack
- **Core:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn
- **NLP:** NLTK or spaCy, TF-IDF, VADER sentiment
- **Modeling:** Logistic Regression (baseline), Random Forest, XGBoost
- **Explainability:** SHAP values, ROC curve, confusion matrix
- **IDE:** VS Code

## Current status
- [x] Dataset loaded and API set up
- [ ] EDA — in progress (`eda.py`)
- [ ] Feature engineering
- [ ] Model training & comparison
- [ ] SHAP explainability
- [ ] (Optional) Streamlit bot score app

## Features to engineer
**Account metadata**
- Follower / following ratio
- Account age in days
- Has profile picture (binary)
- Has bio (binary)
- Verified status
- Tweets per day rate

**Behavioral & text**
- % of tweets that are retweets
- Average tweet sentiment score
- Posting hour variance (bots post at uniform intervals — use std dev of inter-tweet times)
- Unique word ratio (lexical diversity)
- URL presence rate
- Hashtag count per tweet

## Build order
1. EDA — class balance, distributions, visual separation between bot/human
2. Feature engineering — build the columns above
3. Train models — LR baseline → Random Forest → XGBoost
4. Evaluate on F1 + ROC-AUC (not accuracy — classes may be imbalanced)
5. SHAP explainability — beeswarm plot of top bot signals
6. (Optional) NetworkX graph features for follower network analysis

## Target metrics
- ROC-AUC > 0.90
- F1 > 0.85
- 3+ feature types, 3+ models compared

## EDA starter code (already written)
See `eda.py` — loads data, checks class balance, plots distributions by label.
