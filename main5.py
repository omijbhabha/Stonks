import pandas as pd

df=pd.read_csv('spy.csv')
#print(df.head())
from sklearn.model_selection import train_test_split
X=df.drop(columns='Open')
y=df['Close']
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y random_state=8)
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier
e