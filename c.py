import pandas as pd

df_unique = pd.read_csv("unique_merged_final_dataset.csv")
df_processed = pd.read_csv("processed_dataset.csv")

df_unique["userId"] = df_unique["userId"].astype(str)
df_processed["userId"] = df_processed["userId"].astype(str)

soru_sutunlari = [
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

df_sadece_sorular = df_processed[["userId"] + soru_sutunlari].copy()

df_unique.set_index("userId", inplace=True)
df_sadece_sorular.set_index("userId", inplace=True)

df_unique.update(df_sadece_sorular)

df_unique.reset_index().to_csv("updated_unique_merged_final_dataset.csv", index=False)
