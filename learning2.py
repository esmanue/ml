import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("learning2.csv")

target_columns = ["Back", "Chest", "Leg", "Shoulder", "Triceps", "Biceps", "Core", "Cardio"]
question_cols = df.drop(columns=["userId"] + target_columns)  
muscle_cols = df[target_columns]

X = question_cols.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int) #each text ignored
Y = muscle_cols.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
used_columns = X_train.columns 

models = {}
train_maes = {}
test_maes = {}

for col in target_columns:
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train[col])
    models[col] = model

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_maes[col] = mean_absolute_error(y_train[col], train_pred)
    test_maes[col] = mean_absolute_error(y_test[col], test_pred)


print(" MAE Scores (Train vs Test)")
for col in target_columns:
    print(f"{col} | Train MAE: {train_maes[col]:} | Test MAE: {test_maes[col]:}")


for muscle in target_columns:
    model = models[muscle]
    importances = model.feature_importances_
    plt.barh(used_columns, importances)
    plt.title(f"{muscle} için Feature Importance")
    plt.show()