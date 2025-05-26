import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer

def classify_with_faiss(index_type='flat', k=5, top_n=3):
    print(f"\nClassifying using FAISS index: {index_type.upper()} | K = {k}, Top-N = {top_n}")

    test_df = pd.read_csv('outputs/results/test_data.csv')

    # Parse category_multi column
    if 'category_multi' in test_df.columns and isinstance(test_df['category_multi'].iloc[0], str):
        test_df['category_multi'] = test_df['category_multi'].apply(eval)

    # training labels
    labels = np.load(f'outputs/embeddings/train_labels_multi_{index_type}.npy', allow_pickle=True)

    # FAISS index
    index = faiss.read_index(f'outputs/embeddings/faiss_index_{index_type}.index')

    # embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Generate test embeddings
    texts = test_df['product_title'].fillna('') + " " + test_df['about_product'].fillna('')
    print(" Embedding test data...")
    test_embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size].tolist()
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        test_embeddings.append(batch_embeddings)

    test_embeddings = np.vstack(test_embeddings)

    # KNN
    print(" Performing K-NN search...")
    _, indices = index.search(test_embeddings, k)

    # Predict Top-N categories
    predictions = []
    for neighbor_indices in indices:
        all_tags = [tag for idx in neighbor_indices for tag in labels[idx]]
        top_k_tags = [tag for tag, _ in Counter(all_tags).most_common(top_n)]
        predictions.append(top_k_tags)

    # predictions
    test_df['predicted_categories'] = predictions
    output_path = f'outputs/results/test_predictions_{index_type}_top{top_n}.csv'
    test_df.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    classify_with_faiss(index_type='flat', k=5, top_n=2)
