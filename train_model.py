"""
Скрипт, которым обучена и сохранена модель (model.joblib).
Запускать не обязательно — модель уже приложена готовая.
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

df = sns.load_dataset('titanic')
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'
df = df[features + [target]].copy()

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, stratify=df[target], random_state=42
)

numeric_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

preprocess = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
])

model = Pipeline([('prep', preprocess), ('clf', LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)
joblib.dump(model, 'model.joblib')
