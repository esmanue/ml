import json
import csv
from collections import Counter
import re

muscle_id_to_name = {
    "668ceb535e669a8d0a8ba653": "Back",
    "668ceb8e5e669a8d0a8ba654": "Chest",
    "668cebab5e669a8d0a8ba655": "Leg",
    "668cebc15e669a8d0a8ba656": "Shoulder",
    "668cebde5e669a8d0a8ba657": "Triceps",
    "668cebf25e669a8d0a8ba658": "Biceps",
    "668cec0f5e669a8d0a8ba659": "Core",
    "668cec4a5e669a8d0a8ba65b": "Cardio"
}

with open("memberProgramCard.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
muscle_group = list(muscle_id_to_name.values())

for entry in data:
    user_id = entry.get("userId")

    for day in entry.get("programs", []):
        day_name = day.get("DailyProgramName", "Gün ?")
        match = re.search(r"\d+", day_name)
        day_number = int(match.group()) if match else 0  
        counter = Counter()

        for move in day.get("DailyMovements", []):
            muscle_id = move.get("MuscleGroupId")
            muscle_name = muscle_id_to_name.get(muscle_id)
            if muscle_name:
                counter[muscle_name] += 1

        row = [user_id, day_number]
        for muscle in muscle_group:
            row.append(counter.get(muscle, 0))
        rows.append(row)


header = ["user_id", "day"] + muscle_group


with open("musclesnumber.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("musclesnumber.csv başarıyla oluşturuldu.")
