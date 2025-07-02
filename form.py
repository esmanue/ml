import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
import re
from movements import movements_leg, movements_back, movements_chest, movements_core, movements_biceps, movements_cardio, movements_hip, movements_shoulder, movements_triceps

# Load models and columns
day_model = joblib.load("day_prediction_model.pkl")
day_features = joblib.load("day_model_columns.pkl")

muscle_model = joblib.load("muscle_recommendation_model.pkl")
muscle_list = joblib.load("muscle_columns.pkl")

embedder = joblib.load("sentence_embed_model.pkl")
kmeans_q11 = joblib.load("kmeans_q11.pkl")
kmeans_q12 = joblib.load("kmeans_q12.pkl")

data = pd.read_csv("unique_merged_final_dataset.csv")
input_data = data[day_features]

# Movement dictionary
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

# Text preprocessing and classification helpers
def normalize_text(text):
    text = str(text).lower().strip()
    return re.sub(r"[^\w\s]", "", text)

def negativeorpositive(answer: str) -> int:
    answer = normalize_text(answer)
    negatives = ["hayır", "yok", "sıkıntı yok", "problem yok", "hissetmiyorum", "rahatsızlığım yok", "bilincimi kaybetmedim", "düşmedim", "baş dönmesi yok", "ilaç kullanmıyorum"]
    for ref in negatives:
        if fuzz.partial_ratio(answer, ref) > 85:
            return 0
    risks = ["evet", "hissediyorum", "rahatsızlığım", "doktor", "kalp", "göğüs", "ağrı", "bilincimi kaybettim", "ilaç", "düşüyorum"]
    for word in risks:
        if word in answer:
            return 1
    if len(answer) < 3 or answer in ["", "bilinmiyor", "emin değilim", "kararsızım"]:
        return 1
    return 1

def sport_history(answer: str) -> int:
    answer = normalize_text(answer)
    negatives = ["hayır", "yapmadım", "yapmıyorum", "hiç", "yok", "olmadı"]
    if any(neg in answer for neg in negatives): return 0
    if "evet" in answer or re.search(r"\d", answer): return 1
    for sport in ["yüzme", "koşu", "fitness", "yoga", "pilates", "halter", "gym", "tesis", "vücut geliştirme", "bisiklet", "basketbol", "futbol", "voleybol", "tenis"]:
        if sport in answer: return 1
    return 0

def homeorsalon(answer: str) -> int:
    answer = normalize_text(answer)
    if any(kw in answer for kw in ["evde yapmayacağım", "salonda", "spor salonu", "salon", "salonda yapıyorum", "dışarıda", "macfit"]): return 0
    if "evde" in answer or "ev ortamı" in answer: return 1
    return 0

def gym_type(answer: str) -> int:
    answer = normalize_text(answer)
    if "macfit" in answer or "fitpoint" in answer or "kapsamlı" in answer: return 1
    if "evde" in answer or "ev" in answer: return 0
    if "salon" in answer or "gym" in answer or "spor merkezi" in answer: return 2
    return 0

def cluster_from_text(text, embed_model, kmeans_model):
    embedding = embed_model.encode([text])
    return kmeans_model.predict(embedding)[0]

def map_cardio_answer_or_keep(answer):
    answer = normalize_text(answer)
    if any(kw in answer for kw in ["hayır", "yapamam", "zor", "zaman ayıramam", "ayıramam", "olumsuz"]): return 0
    if any(kw in answer for kw in ["evet", "yapabilirim", "olumlu", "mümkün", "uygun"]): return 1
    return 1

# Questions
questions = [
    "1. Baş dönmesi sebebiyle dengenizi kaybediyor musunuz ya da hiç bilincinizi kaybediyor musunuz?",  
    "2. Doktorunuz hiç kalp rahatsızlığınız olduğunu ve sadece bir doktor tarafından önerilen fiziksel aktiviteyi yapmanız gerektiğini söyledi mi?",  
    "3. Doktorunuz şu anda kan basıncınız veya kalp rahatsızlığınız için ilaç kullanmanızı önerdi mi?",  
    "4. Fiziksel aktivite yapmamanız için başka bir neden biliyor musunuz?",                      
    "5. Fiziksel aktiviteyi yaparken göğsünüzde ağrı hissediyor musunuz?",                        
    "6. Son 3 antrenmanınızı hangi tarihlerde yaptınız?:",                                        
    "7. Son bir ay içerisinde fiziksel aktivite yapmadığınız halde göğüs ağrınız oldu mu?",       
    "8. Spor geçmişiniz var mı? Varsa, aktif olarak ne zamandır spor yapıyorsunuz?",              
    "9. Spor salonunuz çok kapsamlı değilse, ekipman görsellerini paylaşabilir misiniz?",        
    "10. Spor yapmanıza engel olacak doktor teşhisli bir rahatsızlığınız varsa, açıklayabilir misiniz?", 
    "11. Sporunuzu evinizde yapacaksanız, ekipmanlarınızla ilgili bilgi verebilir misiniz?",      
    "12. En son uyguladığınız antrenman planlamasını ve sıklığını belirtiniz:", 
    "13. Verilen kardiyo planlamasını gün içinde yapma şansınız var mı?"
]

# User input → binary form
answers_binary = [None] * 13
answers_binary[0] = map_cardio_answer_or_keep(input(questions[12]))               
answers_binary[1] = negativeorpositive(input(questions[1]))                        
answers_binary[2] = negativeorpositive(input(questions[2]))                       
answers_binary[3] = negativeorpositive(input(questions[3]))                        
answers_binary[4] = negativeorpositive(input(questions[4]))                        
answers_binary[5] = cluster_from_text(input(questions[5]), embedder, kmeans_q12)  
answers_binary[6] = negativeorpositive(input(questions[6]))                        
answers_binary[7] = sport_history(input(questions[7]))                             
answers_binary[8] = gym_type(input(questions[8]))                                  
answers_binary[9] = negativeorpositive(input(questions[9]))                        
answers_binary[10] = homeorsalon(input(questions[10]))                             
answers_binary[11] = cluster_from_text(input(questions[11]), embedder, kmeans_q11) 
answers_binary[12] = 0 

# Prediction
input_array = np.array(answers_binary).reshape(1, -1)

day_result = round(day_model.predict(input_array)[0])
print(f"\n Predicted Workout Days: {day_result} days")

muscle_output = muscle_model.predict(input_array)
predicted_muscles = [m for m, val in zip(muscle_list, muscle_output[0]) if val == 1]

matched_row = data[(input_data == answers_binary).all(axis=1)]

if not matched_row.empty:
    movement_text = matched_row.iloc[0]["movements"]
else:
    similarity_scores = input_data.apply(lambda row: sum(row == answers_binary), axis=1)
    best_index = similarity_scores.idxmax()
    movement_text = data.loc[best_index, "movements"]

if pd.notna(movement_text): #for not nan variables
    all_moves = movement_text.split(" | ")
    selected_moves = [m for m in all_moves if any(m.startswith(muscle) for muscle in predicted_muscles)] #seaech for muscle name

    if not selected_moves:
        print("No suitable movements found for predicted muscles.")
        exit()

    plan = defaultdict(list)
    day_now = 1
    move_count = 0
    cardio = any(m.startswith("Cardio") for m in selected_moves)

    total_moves = len(selected_moves)
    moves_per_day = max(1, total_moves // day_result)  #if there is no cardio we look day_result

    for move in selected_moves:
        if day_now > day_result:
            break
        plan[day_now].append(move)

        if cardio:
            if move.startswith("Cardio"):
                day_now += 1
        else:
            move_count += 1
            if move_count % moves_per_day == 0:
                day_now += 1

    # weekly workout plan printer
    print("\nWeekly Workout Plan:")
    for d in range(1, day_result + 1):
        print(f"\nDay {d}:")
        if plan[d]:
            for move in plan[d]:
                print(f"- {move}")
        else:
            print("Rest day or to be planned manually.")
else:
    print("\nNo movement suggestions found.")