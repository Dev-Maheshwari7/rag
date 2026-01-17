from sentence_transformers import SentenceTransformer
from chunking import chunking
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_sentences(sentences):
    # sentences is a list of text chunks, encode them directly
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    return embeddings

def embed_query(sentence):
    embedding = model.encode([sentence])
    return embedding