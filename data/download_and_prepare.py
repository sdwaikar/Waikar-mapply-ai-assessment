import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def download_and_prepare_data():
    os.makedirs('outputs/embeddings', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    print("🌐 Downloading dataset from Hugging Face...")
    dataset = load_dataset("bprateek/amazon_product_description", split="train")

    # Convert to DataFrame
    df = pd.DataFrame({
        'product_title': dataset['Product Name'],
        'about_product': dataset['About Product'],
        'category': dataset['Category']
    })

    # Drop rows with missing category
    df = df.dropna(subset=['category'])
    df['about_product'] = df['about_product'].fillna("")

    # ✅ Split categories into independent tags
    df['category_multi'] = df['category'].apply(lambda x: [c.strip() for c in x.split('|')])

    # ✅ Use first tag for stratification
    df['primary_category'] = df['category_multi'].apply(lambda x: x[0] if x else 'Unknown')

    # ✅ Filter out rare primary categories (only 1 sample)
    valid_primary = df['primary_category'].value_counts()
    df = df[df['primary_category'].isin(valid_primary[valid_primary >= 2].index)]

    # ✅ Stratified 80/20 split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['primary_category']
    )

    # Save train/test
    train_df.to_csv('outputs/results/train_data.csv', index=False)
    test_df.to_csv('outputs/results/test_data.csv', index=False)

    print("\n✅ Data preparation complete!")
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    print(f"Example category_multi: {train_df['category_multi'].iloc[0]}")
    print(f"Unique primary categories: {df['primary_category'].nunique()}")

    return train_df, test_df

if __name__ == "__main__":
    download_and_prepare_data()
