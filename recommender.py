import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



class ContentRecommender:
    def __init__(self, news_df, user_history):
        self.news_df = news_df
        self.user_history = user_history
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.news_vectors = None
        self.news_index = {news_id: idx for idx, news_id in enumerate(news_df['news_id'])}

    def fit(self):
        titles = self.news_df['title'].fillna('')
        self.news_vectors = self.vectorizer.fit_transform(titles)

    def recommend_for_user(self, user_id, top_k=5):
        clicked_ids = self.user_history.get(user_id, [])
        if not clicked_ids:
            return []

        # Get vectors for clicked news
        clicked_idxs = [self.news_index[nid] for nid in clicked_ids if nid in self.news_index]
        if not clicked_idxs:
            return []

        # user_profile = self.news_vectors[clicked_idxs].mean(axis=0)
        user_profile = np.asarray(self.news_vectors[clicked_idxs].mean(axis=0))

        # Compute similarity with all articles
        similarities = cosine_similarity(user_profile, self.news_vectors).flatten()

        # Exclude already clicked articles
        clicked_set = set(clicked_ids)
        recommendations = [
            (self.news_df.iloc[i]['news_id'], similarities[i])
            for i in similarities.argsort()[::-1]
            if self.news_df.iloc[i]['news_id'] not in clicked_set
        ]

        return recommendations[:top_k]
