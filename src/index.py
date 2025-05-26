import os
import time
import faiss
import psutil
import numpy as np

def build_faiss_index(index_type='flat'):
    print(f"\n Building FAISS index: {index_type.upper()}")

    # Loading embeddings and labels
    embeddings = np.load('outputs/embeddings/train_embeddings.npy')
    labels = np.load('outputs/embeddings/train_labels_multi.npy', allow_pickle=True)
    dim = embeddings.shape[1]
    start_time = time.time()
    process = psutil.Process()
    # FAISS index
    if index_type == 'flat':
        index = faiss.IndexFlatL2(dim)
    elif index_type == 'ivf':
        quantizer = faiss.IndexFlatL2(dim)
        nlist = 100
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    else:
        raise ValueError("Unsupported index type. Use 'flat' or 'ivf'.")

    index.add(embeddings)
    indexing_time = time.time() - start_time
    memory_used = process.memory_info().rss / (1024 * 1024)

    # Saving the index and labels
    os.makedirs('outputs/embeddings', exist_ok=True)
    faiss.write_index(index, f'outputs/embeddings/faiss_index_{index_type}.index')
    np.save(f'outputs/embeddings/train_labels_multi_{index_type}.npy',
            np.array(labels, dtype=object))

    with open(f'outputs/embeddings/index_stats_{index_type}.txt', 'w') as f:
        f.write(f"Index type: {index_type}\n")
        f.write(f"Embedding dimension: {dim}\n")
        f.write(f"Vectors indexed: {index.ntotal}\n")
        f.write(f"Indexing time: {indexing_time:.2f} seconds\n")
        f.write(f"Memory usage: {memory_used:.2f} MB\n")

    print(f" Saved: faiss_index_{index_type}.index")
    print(f" Time: {indexing_time:.2f}s | Memory: {memory_used:.2f}MB")

if __name__ == "__main__":
    build_faiss_index('flat')
    build_faiss_index('ivf')
