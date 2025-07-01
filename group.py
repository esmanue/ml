import pandas as pd

df_anket = pd.read_csv("datamerge_Updated_Final.csv")
df_program = pd.read_csv("musclesnumber.csv")


df_sorular = df_anket[["userId"] + list(df_anket.columns[1:14])]

df_birlesik = pd.merge(df_sorular, df_program, on="userId", how="left")

#df_birlesik.to_csv("learning2.csv", index=False)
print(df_birlesik.head())
print("df_program sütunları:", df_program.columns.tolist())

