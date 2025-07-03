import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
import re

day_model = joblib.load("day_prediction_model.pkl")
day_features = joblib.load("day_model_columns.pkl")
embedder = joblib.load("sentence_embed_model.pkl")
kmeans_q11 = joblib.load("kmeans_q11.pkl")
kmeans_q12 = joblib.load("kmeans_q12.pkl")

data = pd.read_csv("final_dataset.csv")
input_data = data[day_features]

questions = [
"1.Doktorunuz hiç kalp rahatsızlığınız olduğunu ve sadece bir doktor tarafından önerilen fiziksel aktiviteyi yapmanız gerektiğini söyledi mi?",
"2.Fiziksel aktiviteyi yaparken göğsünüzde ağrı hissediyor musunuz?",
"3.Son bir ay içerisinde fiziksel aktivite yapmadığınız halde göğüs ağrınız oldu mu?",
"4.Baş dönmesi sebebiyle dengenizi kaybediyor musunuz ya da hiç bilincinizi kaybediyor musunuz?",
"5.Doktorunuz şu anda kan basıncınız veya kalp rahatsızlığınız için ilaç kullanmanızı önerdi mi?",
"6.Fiziksel aktivite yapmamanız için başka bir neden biliyor musunuz?",
"7.Spor yapmanıza engel olacak doktor teşhisli bir rahatsızlığınız varsa, ilgili belgeleri ileterek rahatsızlığınızı açıklayabilir misiniz?",
"8.Spor geçmişiniz var mı? Varsa, aktif olarak ne zamandır spor yapıyorsunuz?",
"9.Sporunuzu evinizde yapacaksanız, bizlere ekipmanlarınızla ilgili görselleri iletebilir misiniz?",
"10.Spor salonunuz çok kapsamlı değilse, ekipman görsellerini paylaşabilir misiniz?",
"11.En son uyguladığınız antrenman planlamasını iletebilir misiniz ve bu planı ne sıklıkla, hangi tarihlerde uyguladığınızı belirtebilir misiniz?",
"12.Son 3 antrenmanınızı hangi tarihlerde yaptınız?",
"13.Verilen kardiyo planlamasını, antrenmandan ayrı bir saatte gün içerisinde yapma şansınız var mı?"
]

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
    if "salon" in answer or "gym" in answer or "spor merkezi" in answer: return 1
    return 0

def cluster_from_text(text, embed_model, kmeans_model):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return -1  
    embedding = embed_model.encode([text])
    return kmeans_model.predict(embedding)[0]


def poz_neg_cardio(answer):
    answer = normalize_text(answer)
    if any(kw in answer for kw in ["hayır", "yapamam", "zor", "zaman ayıramam", "ayıramam", "olumsuz"]): return 0
    if any(kw in answer for kw in ["evet", "yapabilirim", "olumlu", "mümkün", "uygun"]): return 1
    return 1

answers_binary = [None] * 13
answers_binary[0] = negativeorpositive(input(questions[0]))
answers_binary[1] = negativeorpositive(input(questions[1]))
answers_binary[2] = negativeorpositive(input(questions[2]))
answers_binary[3] = negativeorpositive(input(questions[3]))
answers_binary[4] = negativeorpositive(input(questions[4])) 
answers_binary[5] = negativeorpositive(input(questions[5]))
answers_binary[6] = negativeorpositive(input(questions[6]))
answers_binary[7] = sport_history(input(questions[7]))
answers_binary[8] = homeorsalon(input(questions[8]))
answers_binary[9] = gym_type(input(questions[9]))
answers_binary[10] = cluster_from_text(input(questions[10]), embedder, kmeans_q11) 
answers_binary[11] = cluster_from_text(input(questions[11]), embedder, kmeans_q12)
answers_binary[12] = poz_neg_cardio(input(questions[12]))

input_array = np.array(answers_binary).reshape(1, -1)
predicted_days = int(round(day_model.predict(input_array)[0]))
print(f"\n Predicted Days {predicted_days} day")

program_cols = [col for col in data.columns if col.startswith("program")] #select program columnname

best_program = None
for idx, row in data.iterrows():#for every user data
    for prog_col in program_cols: 
        if prog_col in row and pd.notna(row[prog_col]):  #if preg_col not nan
            movements = row[prog_col].split(" | ") #split with |
            cardio_days = sum(1 for move in movements if move.startswith("Cardio")) #each cardio equals one day and increase counter
            if cardio_days >= predicted_days or len(movements) >= predicted_days: # cardio days and movements len control
                best_program = movements
                break
    if best_program:
        break

if not best_program:
    for idx, row in data.iterrows(): 
        for prog_col in program_cols:
            if prog_col in row and pd.notna(row[prog_col]):
                all_moves = row[prog_col].split(" | ")
                if len(all_moves) >= predicted_days:
                    best_program = all_moves
                    break
        if best_program:
            break

plan = defaultdict(list)
day = 1
count = 0
moves_per_day = max(1, len(best_program) // predicted_days)

for move in best_program:
    plan[day].append(move)
    count += 1
    if count % moves_per_day == 0:
        day += 1
    if day > predicted_days:
        break

print("\n Weekly Workout Plan:")
for d in range(1, predicted_days + 1):
    print(f"\nDay {d}:")
    if plan[d]:
        for m in plan[d]:
            print(" -", m)
    else:
        print(" ")
