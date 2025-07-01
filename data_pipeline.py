import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

BATCH_SIZE=100
load_dotenv()

def clean_genres(genre_string):
    if pd.isna(genre_string): return ""
    return ", ".join(g.strip() for g in genre_string.split(","))

chroma_client = chromadb.PersistentClient("./embeddingsDB")
collection = chroma_client.get_or_create_collection(
    name="movie_embeddings",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
)

df = pd.read_csv(os.environ["DATASET_PATH"])
df = df[["movie", "genre", "rating", "description"]]
df_unique = df[~df["movie"].duplicated(keep=False)]
data_dict = df_unique.to_dict(orient="records")

ids = [item["movie"] for item in data_dict]
documents = [item["description"] for item in data_dict]
metadatas = [{"genre": clean_genres(item["genre"]), "rating":item["rating"]} for item in data_dict]


for i in tqdm(range(0, len(ids), BATCH_SIZE)):

    batch_ids = ids[i:i+BATCH_SIZE]
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_meta = metadatas[i:i+BATCH_SIZE]

    try:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )
    except Exception as e:
        print(f"Error: {e} — Batch: {i} - {i+BATCH_SIZE}")
