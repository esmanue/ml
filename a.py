import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
df = pd.read_csv("unique_merged_final_dataset.csv")

def extract_muscles(movement_str): #extracts unique muscle group names from the movement
    if pd.isna(movement_str):
        return []
    return list(set([x.split(":")[0].strip() for x in movement_str.split(" | ") if ":" in x]))

df["muscle_labels"] = df["movements"].apply(extract_muscles)

X = df[[f"q{i}" for i in range(1, 14)]]
y = df["muscle_labels"]

mlb = MultiLabelBinarizer() # Encode the multi-label targets into binary format
y_encoded = mlb.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) # Split into training and testing sets
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "muscle_recommendation_model.pkl")
joblib.dump(mlb.classes_, "muscle_columns.pkl")