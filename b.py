import joblib
import re
from rapidfuzz import fuzz
import pandas as pd

embedder = joblib.load("sentence_embed_model.pkl")
kmeans_q11 = joblib.load("kmeans_q11.pkl")
kmeans_q12 = joblib.load("kmeans_q12.pkl")

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

data=pd.read_csv("merged_user_data.csv")
question_function_map = {
    "Doktorunuz hiç kalp rahatsızlığınız olduğunu ve sadece bir doktor tarafından önerilen fiziksel aktiviteyi yapmanız gerektiğini söyledi mi?": negativeorpositive,
    "Fiziksel aktiviteyi yaparken göğsünüzde ağrı hissediyor musunuz?": negativeorpositive,
    "Son bir ay içerisinde fiziksel aktivite yapmadığınız halde göğüs ağrınız oldu mu?": negativeorpositive,
    "Baş dönmesi sebebiyle dengenizi kaybediyor musunuz ya da hiç bilincinizi kaybediyor musunuz?": negativeorpositive,
    "Doktorunuz şu anda kan basıncınız veya kalp rahatsızlığınız için ilaç kullanmanızı önerdi mi?": negativeorpositive,
    "Fiziksel aktivite yapmamanız için başka bir neden biliyor musunuz?": negativeorpositive,
    "Spor yapmanıza engel olacak doktor teşhisli bir rahatsızlığınız varsa, ilgili belgeleri ileterek rahatsızlığınızı açıklayabilir misiniz?": negativeorpositive,
    "Spor geçmişiniz var mı? Varsa, aktif olarak ne zamandır spor yapıyorsunuz?": sport_history,
    "Sporunuzu evinizde yapacaksanız, bizlere ekipmanlarınızla ilgili görselleri iletebilir misiniz? Ekipmanlarınız mevcut değilse, temin etme durumunuz hakkında olumlu/olumsuz bilgi verebilir misiniz?": homeorsalon,
    "Spor salonunuz çok kapsamlı değilse, ekipman görsellerini paylaşabilir misiniz? Kapsamlı bir salon ise, salonun adını belirtebilir misiniz?": gym_type,
    "En son uyguladığınız antrenman planlamasını iletebilir misiniz ve bu planı ne sıklıkla, hangi tarihlerde uyguladığınızı belirtebilir misiniz?": lambda text: cluster_from_text(text, embedder, kmeans_q11),
    "Son 3 antrenmanınızı hangi tarihlerde yaptınız?": lambda text: cluster_from_text(text, embedder, kmeans_q12),
    "Verilen kardiyo planlamasını, antrenmandan ayrı bir saatte gün içerisinde yapma şansınız var mı? Olumlu/olumsuz olarak belirtebilir misiniz?": poz_neg_cardio,
}


"""processed_data=pd.DataFrame()
processed_data["userId"]=data["userId"]


for question,func in question_function_map.items():
    if question in data.columns:
        colname=question
        processed_data[colname]=data[question].apply(func)

processed_data["program"]=data["program"]

processed_data.to_csv("processed_dataset.csv",index=False)"""

data=pd.read_csv("processed_dataset.csv")

question_columns=[
    "Doktorunuz hiç kalp rahatsızlığınız olduğunu ve sadece bir doktor tarafından önerilen fiziksel aktiviteyi yapmanız gerektiğini söyledi mi?",
    "Fiziksel aktiviteyi yaparken göğsünüzde ağrı hissediyor musunuz?",
    "Son bir ay içerisinde fiziksel aktivite yapmadığınız halde göğüs ağrınız oldu mu?",
    "Baş dönmesi sebebiyle dengenizi kaybediyor musunuz ya da hiç bilincinizi kaybediyor musunuz?",
    "Doktorunuz şu anda kan basıncınız veya kalp rahatsızlığınız için ilaç kullanmanızı önerdi mi?",
    "Fiziksel aktivite yapmamanız için başka bir neden biliyor musunuz?",
    "Spor yapmanıza engel olacak doktor teşhisli bir rahatsızlığınız varsa, ilgili belgeleri ileterek rahatsızlığınızı açıklayabilir misiniz?",
    "Spor geçmişiniz var mı? Varsa, aktif olarak ne zamandır spor yapıyorsunuz?",
    "Sporunuzu evinizde yapacaksanız, bizlere ekipmanlarınızla ilgili görselleri iletebilir misiniz? Ekipmanlarınız mevcut değilse, temin etme durumunuz hakkında olumlu/olumsuz bilgi verebilir misiniz?",
    "Spor salonunuz çok kapsamlı değilse, ekipman görsellerini paylaşabilir misiniz? Kapsamlı bir salon ise, salonun adını belirtebilir misiniz?",
    "En son uyguladığınız antrenman planlamasını iletebilir misiniz ve bu planı ne sıklıkla, hangi tarihlerde uyguladığınızı belirtebilir misiniz?",
    "Son 3 antrenmanınızı hangi tarihlerde yaptınız?",
    "Verilen kardiyo planlamasını, antrenmandan ayrı bir saatte gün içerisinde yapma şansınız var mı? Olumlu/olumsuz olarak belirtebilir misiniz?"
]

import pandas as pd

processed_data = pd.read_csv("processed_dataset.csv")
question_columns = [col for col in processed_data.columns if col not in ["userId", "program"]]
grouped = processed_data.groupby(question_columns)["program"].apply(list).reset_index()
def get_first_5_programs(program_list):
    first5 = program_list[:5]
    while len(first5) < 5:
        first5.append(None)
    return pd.Series(first5)


program_df = grouped["program"].apply(get_first_5_programs)
program_df.columns = [f"program{i+1}" for i in range(5)]

final_result = pd.concat([grouped[question_columns], program_df], axis=1)

final_result.to_csv("sorulara_gore_5_tam_program.csv", index=False)
 








