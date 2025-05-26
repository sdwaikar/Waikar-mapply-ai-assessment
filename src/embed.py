import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def generate_embeddings():
    print("Loading training data...")
    train_df = pd.read_csv('outputs/results/train_data.csv')

    if isinstance(train_df['category_multi'].iloc[0], str):
        train_df['category_multi'] = train_df['category_multi'].apply(eval)

    # embedding model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = train_df['product_title'].fillna('') + " " + train_df['about_product'].fillna('')
    print(" Generating embeddings...")
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size].tolist()
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    os.makedirs('outputs/embeddings', exist_ok=True)
    np.save('outputs/embeddings/train_embeddings.npy', embeddings)
    np.save('outputs/embeddings/train_labels_multi.npy',
            np.array(train_df['category_multi'].tolist(), dtype=object))

    with open('outputs/embeddings/model_info.txt', 'w') as f:
        f.write(f"Model used: {model_name}\n")
        f.write("Input: product_title + about_product\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n")
        f.write(f"Samples: {len(embeddings)}\n")

    print(" Embedding generation complete!")

if __name__ == "__main__":
    generate_embeddings()
