import pandas as pd
from src.preprocessing import ensure_columns
def test_ensure_columns_basic():
    df=pd.DataFrame({'pclass':[1],'sex':['male'],'age':[22],'fare':[7.25],'embarked':['S'],'SibSp':[1],'Parch':[0]})
    out=ensure_columns(df)
    for col in ['Pclass','Sex','Age','Fare','Embarked','SibSp','Parch']:
        assert col in out.columns