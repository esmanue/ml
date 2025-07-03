import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import joblib
df = pd.read_csv("final_dataset.csv")

program1= ["program1","program2","program3","program4","program5"]
program_edu=["program1"]
question_col=[col for col in df.columns if col not in program1 and col!="userId"]

df["movements"] = df["program1"]

def extract_muscles(movement_str): #extracts unique muscle group names from the movement
    if pd.isna(movement_str):
        return []
    return list(set([x.split(":")[0].strip() for x in movement_str.split(" | ") if ":" in x]))

df["muscle_labels"] = df["movements"].apply(extract_muscles)

X = df[question_col]
y = df["muscle_labels"]

mlb = MultiLabelBinarizer() # Encode the multi-label targets into binary format
y_encoded = mlb.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) # Split into training and testing sets
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=mlb.classes_))

joblib.dump(model, "muscle_recommendation_model.pkl")
joblib.dump(mlb.classes_, "muscle_columns.pkl")