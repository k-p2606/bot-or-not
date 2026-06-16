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

## Bot vs Human Twitter/X Classifier

**Project summary:** Train a classifier to distinguish bot accounts from real humans on Twitter/X using behavioral patterns, account metadata, and tweet language features — then surface what signals are most diagnostic of bot activity.

**Why this project:** Bot detection is a real, active problem at every major tech company (Twitter, Meta, Google, TikTok). It combines NLP, behavioral analytics, and anomaly detection — interviewers recognize it as production-relevant work.

---

### Dataset

> ⚠️ X's API is now heavily paywalled. Use pre-labeled datasets instead.

- **TwiBot-22** *(recommended)* — 1.1M accounts, labeled human/bot, includes tweets, follower graphs, and metadata. Academic benchmark, cite in README.
- **Kaggle bot datasets** — ~37k accounts, easier to start with. Search "Twitter bot detection dataset" on Kaggle.

---

### Tech Stack

| Area | Tools |
|---|---|
| Core | Python, pandas, numpy, scikit-learn, matplotlib, seaborn |
| NLP | NLTK or spaCy, TF-IDF, VADER or TextBlob for sentiment |
| Modeling | Logistic Regression, Random Forest, XGBoost; optional BERT fine-tune |
| Explainability | SHAP, LIME, confusion matrix, ROC curve |

---

### Features to Engineer

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
- Posting hour variance (bots post uniformly)
- Unique word ratio in tweets
- URL presence rate
- Hashtag count per tweet

---

### Build Steps

1. **Load & explore the data** — EDA on bot/human class split, distributions of follower ratios, account ages, tweet rates. These visualizations tell a compelling story on their own.

2. **Engineer behavioral features** — Posting hour variance is the most interesting: bots post at mathematically uniform intervals, humans are erratic. Calculate std dev of posting times per account.

3. **Add NLP features from tweet text** — TF-IDF on tweet content for vocabulary patterns. Lexical diversity (unique words / total words) — bots reuse phrases. VADER sentiment to check for unnaturally uniform tone.

4. **Train & compare models** — Logistic Regression as interpretable baseline → Random Forest → XGBoost. Evaluate on F1 and ROC-AUC (accuracy is misleading with class imbalance). Use stratified k-fold CV.

5. **SHAP explainability** — Beeswarm plot showing that "tweets per day > 50" and "follower ratio < 0.01" are the strongest bot signals. Great portfolio visualization.

6. **Bot score app (optional)** — Streamlit app with manual input fields that returns a bot probability score with top contributing features.

---

### Advanced Extension

Build a follower graph with **NetworkX**. Bots often follow each other in clusters and have star-shaped networks. Add graph-based features like clustering coefficient and degree centrality to push accuracy higher and signal serious technical depth.

---

### Target Metrics

| Metric | Target |
|---|---|
| ROC-AUC | > 0.90 |
| F1 score | > 0.85 |
| Feature types | 3+ |
| Models compared | 3+ |

---

### Resume Bullets

- Built a bot detection classifier on 37,000+ labeled Twitter accounts, engineering 14 behavioral and NLP features including posting time variance, lexical diversity, and follower ratio, achieving 0.91 ROC-AUC with XGBoost.
- Designed a posting cadence feature measuring standard deviation of inter-tweet intervals, identifying that bot accounts post with 3.2× more temporal uniformity than human accounts.
- Applied SHAP explainability to surface the top 5 bot signals, translating black-box model outputs into interpretable rules actionable for platform trust and safety teams.
- Extended the model with graph-based features using NetworkX to compute follower network clustering coefficients, improving F1 by 8% over metadata-only baseline.