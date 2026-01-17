import faiss   
def create_faiss_index(d, embeddings):           
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(embeddings)
    return index

#d=384 for all-MiniLM-L6-v2
if __name__ == "__main__":
    pass