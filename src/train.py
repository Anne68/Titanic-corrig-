import os, joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .logger import logger
from .preprocessing import build_preprocess_pipeline, ALL_FEATURES, ensure_columns
DATA_PATH = os.getenv("DATA_PATH", "data/titanic_train.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, os.getenv("MODEL_NAME", "model.pkl"))
def main():
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = ensure_columns(df)
    y = df["Survived"].astype(int)
    X = df[ALL_FEATURES]
    pre = build_preprocess_pipeline()
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre),("clf", clf)])
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    pipe.fit(X_train,y_train)
    acc=accuracy_score(y_val, pipe.predict(X_val))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH} | val_accuracy={acc:.4f}")
if __name__=='__main__': main()