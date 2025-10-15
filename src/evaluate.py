import pandas as pd, joblib, mlflow
from sklearn.metrics import classification_report, accuracy_score, f1_score

def eval_one(name, path, test_csv='data/test.csv'):
    df = pd.read_csv(test_csv); X, y = df['text'].astype(str), df['sentiment']
    pipe = joblib.load(path); yhat = pipe.predict(X)
    print(f"\n--- {name} ---")
    print(classification_report(y, yhat, target_names=['Négatif','Positif']))
    acc = accuracy_score(y, yhat); f1w = f1_score(y, yhat, average='weighted')
    print(f"Accuracy: {acc:.4f} | F1-pondéré: {f1w:.4f}")
    with mlflow.start_run(run_name=f"Eval - {name}"):
        mlflow.log_metric("accuracy", acc); mlflow.log_metric("f1_weighted", f1w)
    return acc, f1w

if __name__ == "__main__":
    eval_one("LogisticRegression", "models/logisticregression_pipeline.joblib")
    eval_one("Naive Bayes", "models/naive_bayes_pipeline.joblib")
