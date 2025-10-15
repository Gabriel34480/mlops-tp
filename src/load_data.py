import pandas as pd
import requests, zipfile, io, os

def load_and_prepare_data(url, data_dir='data', sample_n=50000, random_state=42):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv')
    if not os.path.exists(csv_path):
        print("Téléchargement du jeu de données...")
        r = requests.get(url, timeout=120); r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(data_dir)
        print("Téléchargement et extraction terminés.")
    cols = ['sentiment','id','date','query','user','text']
    df = pd.read_csv(csv_path, header=None, names=cols, encoding='latin-1', on_bad_lines='skip')
    df = df[['sentiment','text']]
    df['sentiment'] = df['sentiment'].replace({4:1})
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)
    return df

if __name__ == "__main__":
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    df = load_and_prepare_data(url, sample_n=50000)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_tweets.csv', index=False)
    print("Échantillon sauvegardé dans data/raw_tweets.csv")
