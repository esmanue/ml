from rapidfuzz import fuzz
import re
import joblib
import numpy as np
from movements import movements_leg, movements_back,movements_chest, movements_core, movements_biceps,movements_cardio,movements_hip,movements_shoulder,movements_triceps
from collections import defaultdict
import random

model = joblib.load("exercise_recommendation_model.pkl") # Load the trained model and column name
valid_cols = joblib.load("exercise_columns.pkl")

# movement-muscle converter 
all_movement_lists = {
    "Biceps": movements_biceps,
    "Cardio": movements_cardio,
    "Chest": movements_chest,
    "Core": movements_core,
    "Hip": movements_hip,
    "Leg": movements_leg,
    "Shoulder": movements_shoulder,
    "Triceps": movements_triceps,
    "Back": movements_back,
}

movement_to_muscle = {} # create movement muscle dictionary
for muscle, moves in all_movement_lists.items():
    for item in moves: #search for move
        name = item.split(":")[0].strip().upper()
        movement_to_muscle[name] = muscle #match


def normalize_text(text):
    text = str(text).lower().strip() #lowercase, remove unnecessary spaces 
    text = re.sub(r"[^\w\s]", "", text) #remove punctuation
    return text


def negativeorpositive(answer: str) -> int: # turning str to int
    answer = normalize_text(answer)

    negative_words = [
        "hayır", "yok", "sıkıntı yok", "problem yok", "hissetmiyorum", # If the answer matches negative words  return 0
        "rahatsızlığım yok", "bilincimi kaybetmedim", "düşmedim", 
        "baş dönmesi yok", "ilaç kullanmıyorum"
    ]
    for ref in negative_words:
        if fuzz.partial_ratio(answer, ref) > 85:
            return 0

   
    risk_keywords = [
        "evet", "hissediyorum", "rahatsızlığım", "doktor", "kalp",  # If it matches risk words  return 1
        "göğüs", "ağrı", "bilincimi kaybettim", "ilaç", "düşüyorum"
    ]
    for word in risk_keywords:
        if word in answer:
            return 1

    if len(answer) < 3 or answer in ["", "bilinmiyor", "emin değilim", "kararsızım"]:
        return 1

    return 1


def sport_history(answer: str) -> int:
    answer = str(answer).lower().strip()

    negatives = ["hayır", "yapmadım", "yapmıyorum", "hiç", "yok", "olmadı"]
    for neg in negatives:
        if neg in answer:
            return 0

    if "evet" in answer:
        return 1

    if re.search(r"\d", answer): #for numbers
        return 1

    sports = [
        "yüzme", "koşu", "fitness", "yoga", "pilates", "halter", "gym", 
        "tesis", "vücut geliştirme", "bisiklet", "basketbol", "futbol", 
        "voleybol", "tenis"
    ]
    for sport in sports:
        if sport in answer:
            return 1

    return 0


def homeorsalon(answer: str) -> int: # Detect if training is at home
    answer = str(answer).lower().strip()

    not_home_keywords = ["evde yapmayacağım", "salonda", "spor salonu", "salon", "salonda yapıyorum", "dışarıda", "macfit"]
    for phrase in not_home_keywords:
        if phrase in answer:
            return 0

    if "evde" in answer or "ev ortamı" in answer or "evde yapacağım" in answer:
        return 1

    return 0

def gym_type(answer: str) -> int:
    answer = str(answer).lower().strip()

    if "macfit" in answer or "fitpoint" in answer or "kapsamlı" in answer: 
        return 1

    if "evde" in answer or "ev" in answer:
        return 0

    if "salon" in answer or "gym" in answer or "spor merkezi" in answer:
        return 2

    return 0

# for load we use joblib
embedder = joblib.load("sentence_embed_model.pkl") # Load sentence embedding and clustering models 
kmeans_q11 = joblib.load("kmeans_q11.pkl")
kmeans_q12 = joblib.load("kmeans_q12.pkl")

def cluster_from_text(text, embed_model, kmeans_model): #for q11 and q12 convert embedding and clustering
    embedding = embed_model.encode([text])
    cluster = kmeans_model.predict(embedding)[0]
    return cluster

def map_cardio_answer_or_keep(answer):
    answer_lower = str(answer).lower().strip()

    negatives = ["hayır", "yapamam", "zor", "zaman ayıramam", "ayıramam", "olumsuz"]
    positives = ["evet", "yapabilirim", "olumlu", "mümkün", "uygun"]

    for neg in negatives:
        if neg in answer_lower:
            return 0

    for pos in positives:
        if pos in answer_lower:
            return 1

    return 1  # Keep as-is if uncertain

# Questions list
questions = [
    "1. Doktorunuz hiç kalp rahatsızlığınız olduğunu ve sadece bir doktor tarafından önerilen fiziksel aktiviteyi yapmanız gerektiğini söyledi mi?",
    "2. Fiziksel aktiviteyi yaparken göğsünüzde ağrı hissediyor musunuz?",
    "3. Son bir ay içerisinde fiziksel aktivite yapmadığınız halde göğüs ağrınız oldu mu?",
    "4. Baş dönmesi sebebiyle dengenizi kaybediyor musunuz ya da hiç bilincinizi kaybediyor musunuz?",
    "5. Doktorunuz şu anda kan basıncınız veya kalp rahatsızlığınız için ilaç kullanmanızı önerdi mi?",
    "6. Fiziksel aktivite yapmamanız için başka bir neden biliyor musunuz?",
    "7. Spor yapmanıza engel olacak doktor teşhisli bir rahatsızlığınız varsa, açıklayabilir misiniz?",
    "8. Spor geçmişiniz var mı? Varsa, aktif olarak ne zamandır spor yapıyorsunuz?",
    "9. Sporunuzu evinizde yapacaksanız, ekipmanlarınızla ilgili bilgi verebilir misiniz?",
    "10. Spor salonunuz çok kapsamlı değilse, ekipman görsellerini paylaşabilir misiniz?",
    "11. En son uyguladığınız antrenman planlamasını ve sıklığını belirtiniz: ",
    "12. Son 3 antrenmanınızı hangi tarihlerde yaptınız?: ",
    "13. Verilen kardiyo planlamasını, antrenmandan ayrı bir saatte gün içerisinde yapma şansınız var mı? Olumlu/olumsuz olarak belirtebilir misiniz?"
    "14. Kaç günlük bir program istiyorsunuz"
]

# Prepare answers from user input
answers_binary = []
for q in questions[:7]:
    cevap = input(q + " ")
    binary = negativeorpositive(cevap)
    answers_binary.append(binary)

cevap8 = input(questions[7] + " ")
answers_binary.append(sport_history(cevap8))

cevap9 = input(questions[8] + " ")
answers_binary.append(homeorsalon(cevap9))

cevap10 = input(questions[9] + " ")
answers_binary.append(gym_type(cevap10))

cevap11 = input(questions[10] + " ")
answers_binary.append(cluster_from_text(cevap11, embedder, kmeans_q11))

cevap12 = input(questions[11] + " ")
answers_binary.append(cluster_from_text(cevap12, embedder, kmeans_q12))

cevap13 = input(questions[12] + " ")
answers_binary.append(map_cardio_answer_or_keep(cevap13))

cevap14 = input(questions[13]+" ")

# Convert to model input shape
X_input = np.array(answers_binary).reshape(1, -1)
prediction = model.predict(X_input)[0]

print(" binary vector:")
print(prediction)

# Show exercise names 
recommended_exercises = [name for name, val in zip(valid_cols, prediction) if val == 1]

print(" Recommended Exercises:")
for move in recommended_exercises:
    print("-", move)

day_muscle_map = {
    1: ["Chest", "Shoulder"],
    2: ["Back", "Biceps"],
    3: ["Leg", "Shoulder"],
    4: ["Back","Core"],
    5: ["Leg","Core"]
}

muscle_grouped = defaultdict(list) #default dict
for move in recommended_exercises: 
    key = move.strip().upper()
    if key in movement_to_muscle: #search in dictionary 
        muscle = movement_to_muscle[key] #match muscle-movement
        for item in all_movement_lists[muscle]:
            if item.startswith(key): # for set and reps
                muscle_grouped[muscle].append(item)
                break

days = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}

for day, muscles in day_muscle_map.items():
    for muscle in muscles:
        if muscle in muscle_grouped:
            days[day][muscle].extend(muscle_grouped[muscle])


for day in range(1, (int(cevap14)+1)):
    print(f"\nGün {day} ")
    for muscle, moves in days[day].items():
        print(f" {muscle}:")
        for move in moves:
            print("-", move)
