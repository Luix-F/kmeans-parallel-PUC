import pandas as pd

# Carregar apenas as colunas necessárias
df = pd.read_csv(
    "yellow_tripdata_2015-01.csv",
    usecols=["pickup_longitude", "pickup_latitude"]
)

# 1. Remover nulos
df = df.dropna()

# 2. Remover coordenadas (0,0)
df = df[(df["pickup_longitude"] != 0) & (df["pickup_latitude"] != 0)]

# Salvar o arquivo
df.to_csv("data.csv", index=False)

print(f"Processamento concluído. {len(df)} registros válidos salvos.")