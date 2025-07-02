import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


df = pd.read_csv("unique_merged_final_dataset.csv")  

X = df[[f"q{i}" for i in range(1, 14)]]
y = df["total_days"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42) # Create and train a random forest regressor
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate error metrics and mae
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f" Ortalama Hata (MAE): {mae:.2f}")
print(f" R2 Skoru: {r2:.2f}")

joblib.dump(model, "day_prediction_model.pkl")
joblib.dump(X.columns.tolist(), "day_model_columns.pkl")
