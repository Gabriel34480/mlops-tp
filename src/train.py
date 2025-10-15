import os, pandas as pd, mlflow, mlflow.sklearn, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

mlflow.set_experiment("Analyse de Sentiments Twitter")

def make_lr(): return Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
                                ("clf", LogisticRegression(max_iter=200))])
def make_nb(): return Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,1))),
                                ("clf", MultinomialNB())])

def train_and_log(name, pipe, train_csv='data/train.csv'):
    df = pd.read_csv(train_csv); X, y = df['text'].astype(str), df['sentiment']
    with mlflow.start_run(run_name=name):
        p = pipe.get_params()
        mlflow.log_param("model_type", name)
        mlflow.log_param("tfidf_max_features", p['tfidf__max_features'])
        mlflow.log_param("tfidf_ngram_range", p['tfidf__ngram_range'])
        if name=="LogisticRegression": mlflow.log_param("clf_max_iter", p['clf__max_iter'])
        pipe.fit(X, y)
        os.makedirs("models", exist_ok=True)
        path = f"models/{name.lower().replace(' ','_')}_pipeline.joblib"
        joblib.dump(pipe, path); mlflow.log_artifact(path)
        mlflow.sklearn.log_model(pipe, artifact_path=f"{name}_pipeline")
        print(f"{name} entraîné : {path}")

if __name__ == "__main__":
    train_and_log("LogisticRegression", make_lr())
    train_and_log("Naive Bayes", make_nb())
