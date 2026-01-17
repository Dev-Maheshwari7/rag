def chunking(data, chunk_size):
    overlap = max(1, chunk_size // 10)  # 10% overlap
    step = chunk_size - overlap

    for i in range(0, len(data), step):
        yield data[i:i + chunk_size]

if __name__ == "__main__":
    sample_text = "This is a sample text to demonstrate chunking functionality. "
    chunk_size = 10
    chunks = list(chunking(sample_text, chunk_size))
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")