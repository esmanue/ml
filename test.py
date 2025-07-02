import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from movements import movements_cardio  # Cardio listesini getiriyoruz

# Load models and columns
day_model = joblib.load("day_prediction_model.pkl")
day_features = joblib.load("day_model_columns.pkl")

muscle_model = joblib.load("muscle_recommendation_model.pkl")
muscle_list = joblib.load("muscle_columns.pkl")


data = pd.read_csv("unique_merged_final_dataset.csv")
input_data = data[day_features]

# Sample user input
user_input = [0,1,1,0,0,1,1,0,0,0,1,0,1]
input_array = np.array(user_input).reshape(1, -1)


day_result = round(day_model.predict(input_array)[0])# Predict number of workout days
print(f"\nPredicted Workout Days: {day_result} days")


muscle_output = muscle_model.predict(input_array) # Predict muscle groups
predicted_muscles = [m for m, val in zip(muscle_list, muscle_output[0]) if val == 1]


matched_row = data[(input_data == user_input).all(axis=1)]# Try to find exact user match

if not matched_row.empty:
    movement_text = matched_row.iloc[0]["movements"]
else:
    similarity_scores = input_data.apply(lambda row: sum(row == user_input), axis=1)
    best_index = similarity_scores.idxmax()
    movement_text = data.loc[best_index, "movements"]


if pd.notna(movement_text): # If we found movements
    all_moves = movement_text.split(" | ")
    selected_moves = [m for m in all_moves if any(m.startswith(muscle) for muscle in predicted_muscles)]

    plan = defaultdict(list)
    day_now = 1
    move_count = 0
    cardio = any(m.startswith("Cardio") for m in selected_moves)

    for move in selected_moves:
        if day_now > day_result:
            break
        plan[day_now].append(move)

        if cardio:
            if move.startswith("Cardio"):
                day_now += 1
        else:
            move_count += 1
            if move_count % 7 == 0:
                day_now += 1

    """# add cardio if there is no
    for d in range(1, day_result + 1):
        if not any("Cardio" in m for m in plan[d]):
            random_cardio = random.choice(movements_cardio)
            plan[d].append(random_cardio)"""

    # Print the weekly plan
    print("\nWeekly Workout Plan:")
    for d in range(1, day_result + 1):
        print(f"\nDay {d}:")
        if plan[d]:
            for i, move in enumerate(plan[d]):
                dash = "-" 
                print(f"{dash} {move}")
        else:
            print("Rest day or to be planned manually.")
else:
    print("\nNo movement suggestions found.")
