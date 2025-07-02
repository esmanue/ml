import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("merged_muscle_freq.csv")

X = df[[f"q{i}" for i in range(1, 14)]]
y = df[["Chest", "Back", "Shoulder", "Core", "Cardio", "Biceps", "Triceps", "Leg", "Hip"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Ortalama MAE: {mae:.2f}")

new_user = [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 3, 1]]
kas_tahmini = model.predict(new_user)[0]

kas_gruplari = ["Chest", "Back", "Shoulder", "Core", "Cardio", "Biceps", "Triceps", "Leg", "Hip"]
for kas, adet in zip(kas_gruplari, kas_tahmini):
    print(f"{kas}: {round(adet)} hareket")
