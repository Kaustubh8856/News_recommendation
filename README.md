# 📰 Personalized News Recommender (MIND Dataset)

A content-based recommender system that suggests news articles tailored to each user's reading history using the Microsoft MIND dataset.

---

## 🚀 Features

- Personalized news recommendations for each user
- User click history modeling
- TF-IDF + Cosine Similarity based article ranking
- Clean modular code for easy experimentation

---

## 🧠 How It Works

- Parses user click logs and news content
- Builds a profile for each user based on previously clicked articles
- Transforms news content into TF-IDF vectors
- Calculates similarity scores to recommend top articles for each user

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- TF-IDF Vectorization
- Microsoft MIND Dataset

---

## ▶️ How to Run

1. Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/MIND-News-Recommender.git
cd MIND-News-Recommender
pip install -r requirements.txt
python src/preprocess.py
python main.py
```
##📂 Dataset:
- Microsoft News Dataset (MIND-small)
- Download from: https://msnews.github.io
