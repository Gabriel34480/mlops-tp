import pandas as pd, re, nltk, os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

for pkg in [('stopwords','stopwords'), ('punkt','tokenizers/punkt'), ('wordnet','corpora/wordnet')]:
    try:
        if pkg[0]=='stopwords': stopwords.words('english')
        else: nltk.data.find(pkg[1])
    except LookupError:
        nltk.download(pkg[0])

def preprocess_text(text:str)->str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    sw = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in sw and t.strip()!='']
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(w) for w in tokens]
    return " ".join(tokens)

if __name__ == "__main__":
    df = pd.read_csv('data/raw_tweets.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    X, y = df['cleaned_text'], df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs('data', exist_ok=True)
    pd.DataFrame({'text':X_train, 'sentiment':y_train}).to_csv('data/train.csv', index=False)
    pd.DataFrame({'text':X_test, 'sentiment':y_test}).to_csv('data/test.csv', index=False)
    print("Train/Test sauvegardés.")
