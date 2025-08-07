from src.preprocess import load_news, load_behaviors, get_click_history
from src.recommender import ContentRecommender
import random

def main():
    print("🔄 Loading data...")
    news_df = load_news()
    behaviors_df = load_behaviors()
    user_history = get_click_history(behaviors_df)

    print("✅ Data loaded. Training recommender...")
    recommender = ContentRecommender(news_df, user_history)
    recommender.fit()

    # sample_user = list(user_history.keys())[0] --- fixed sample user
    sample_user = random.choice(list(user_history.keys()))  # random sample user
    print(f"\n🔍 Sample user: {sample_user}")
    
    recommendations = recommender.recommend_for_user(sample_user, top_k=5)

    print("\n📢 Top 5 Recommendations:")
    for news_id, score in recommendations:
        title = news_df[news_df['news_id'] == news_id]['title'].values[0]
        print(f"- {title} (ID: {news_id}, Score: {score:.4f})")

if __name__ == '__main__':
    main()
