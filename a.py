import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("datamerge_Updated_Final.csv")

question_cols = df.columns[1:14]
exercise_cols = df.columns[14:]

X = df[question_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int) #each text ignored
Y = df[exercise_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

valid_cols = [col for col in y_train.columns if y_train[col].nunique() > 1]
y_train = y_train[valid_cols]
y_test = y_test[valid_cols]

model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=valid_cols))

joblib.dump(model, "exercise_recommendation_model.pkl")
joblib.dump(valid_cols, "exercise_columns.pkl")