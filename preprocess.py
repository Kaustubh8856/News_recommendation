import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = 'data'

def load_news():
    news_path = os.path.join(DATA_DIR, 'news.tsv')
    columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    
    news_df = pd.read_csv(news_path, sep='\t', names=columns, encoding='utf-8')
    news_df.drop_duplicates(subset='news_id', inplace=True)
    
    return news_df

def load_behaviors():
    behaviors_path = os.path.join(DATA_DIR, 'behaviors.tsv')
    columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    
    behaviors_df = pd.read_csv(behaviors_path, sep='\t', names=columns)
    
    # Fill missing history with empty string
    behaviors_df['history'] = behaviors_df['history'].fillna('')
    
    return behaviors_df

def parse_impressions(behaviors_df):
    """Returns a dataframe of individual impression entries (user, news_id, clicked label)."""
    records = []
    for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df)):
        user = row['user_id']
        impressions = row['impressions'].split()
        
        for item in impressions:
            news_id, clicked = item.split('-')
            records.append({
                'user_id': user,
                'news_id': news_id,
                'clicked': int(clicked)
            })

    return pd.DataFrame(records)

def get_click_history(behaviors_df):
    """Returns a dictionary mapping user_id -> list of clicked news_ids (from history)."""
    user_history = {}
    for _, row in behaviors_df.iterrows():
        user = row['user_id']
        history = row['history'].split()
        if history:
            user_history[user] = history
    return user_history

if __name__ == '__main__':
    news_df = load_news()
    behaviors_df = load_behaviors()

    print("Loaded News Articles:", len(news_df))
    print("Loaded Behavior Entries:", len(behaviors_df))

    impression_df = parse_impressions(behaviors_df)
    print("Parsed Impressions:", len(impression_df))

    user_history = get_click_history(behaviors_df)
    print("Users with history:", len(user_history))
