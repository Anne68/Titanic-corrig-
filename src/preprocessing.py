import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
NUM_COLS = ["Age","Fare","SibSp","Parch"]
CAT_COLS = ["Sex","Embarked","Pclass"]
ALL_FEATURES = CAT_COLS + NUM_COLS
def build_preprocess_pipeline():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, NUM_COLS),("cat", cat_pipe, CAT_COLS)])
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "pclass" in df.columns and "Pclass" not in df.columns: df = df.rename(columns={"pclass":"Pclass"})
    if "sex" in df.columns and "Sex" not in df.columns: df = df.rename(columns={"sex":"Sex"})
    if "age" in df.columns and "Age" not in df.columns: df = df.rename(columns={"age":"Age"})
    if "fare" in df.columns and "Fare" not in df.columns: df = df.rename(columns={"fare":"Fare"})
    if "embarked" in df.columns and "Embarked" not in df.columns: df = df.rename(columns={"embarked":"Embarked"})
    if "Pclass" in df.columns: df["Pclass"] = df["Pclass"].astype("Int64")
    return df