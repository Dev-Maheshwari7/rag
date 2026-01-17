import faiss
def search(query_embedding, index, top_k):
    import numpy as np
    k = top_k                          # we want to see top_k nearest neighbors

    D, I = index.search(query_embedding, k)     # actual search
    return I          # neighbors of the 5 last queries
        # neighbors of the 5 last queries