import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import joblib


df = pd.read_csv("responsesText_binary.csv")


text_q11 = df.iloc[:, 11].fillna("").astype(str)
text_q12 = df.iloc[:, 12].fillna("").astype(str)

embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings_q11 = embed_model.encode(text_q11.tolist())
embeddings_q12 = embed_model.encode(text_q12.tolist())

n_clusters = 3
kmeans_q11 = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_q11)
kmeans_q12 = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_q12)


df.iloc[:, 11] = kmeans_q11.predict(embeddings_q11)
df.iloc[:, 12] = kmeans_q12.predict(embeddings_q12)


df.to_csv("clustered_responses.csv", index=False)

joblib.dump(kmeans_q11, "kmeans_q11.pkl")
joblib.dump(kmeans_q12, "kmeans_q12.pkl")
joblib.dump(embed_model, "sentence_embed_model.pkl")

print("✅ Clustering tamamlandı ve modeller kaydedildi.")
