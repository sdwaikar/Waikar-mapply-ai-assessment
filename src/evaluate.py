import os
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(y_true, y_pred):
    match_count = 0
    for true_tags, pred_tags in zip(y_true, y_pred):
        if all(tag in true_tags for tag in pred_tags):
            match_count += 1
    return match_count / len(y_true)

def evaluate_multilabel(index_type='flat', top_n=2):
    print(f"\nEvaluating Top-{top_n} majority-vote predictions...")

    file_path = f'outputs/results/test_predictions_{index_type}_top{top_n}.csv'
    df = pd.read_csv(file_path)

    # Convert to Python lists
    df['category_multi'] = df['category_multi'].apply(eval)
    df['predicted_categories'] = df['predicted_categories'].apply(eval)

    y_true = df['category_multi']
    y_pred = df['predicted_categories']

    acc = accuracy(y_true, y_pred)

    # Binarize
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    precision = precision_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)

    # metrics
    os.makedirs('outputs/results', exist_ok=True)
    with open(f'outputs/results/eval_metrics_{index_type}_top{top_n}.txt', 'w') as f:
        f.write(f"Accuracy (all predicted in true): {acc:.4f}\n")
        f.write(f"Samples Precision: {precision:.4f}\n")
        f.write(f"Samples Recall: {recall:.4f}\n")
        f.write(f"Samples F1-score: {f1:.4f}\n")

    print(f"Evaluation metrics saved to eval_metrics_{index_type}_top{top_n}.txt")

    # Flattened tag lists
    flat_true = [tag for sublist in y_true for tag in sublist]
    flat_pred = [tag for sublist in y_pred for tag in sublist]

    # Top-20 Tags
    top_tags = [tag for tag, _ in Counter(flat_true).most_common(20)]
    flat_true_filtered = [tag if tag in top_tags else "Other" for tag in flat_true]
    flat_pred_filtered = [tag if tag in top_tags else "Other" for tag in flat_pred]

    # confusion matrix
    cm = pd.crosstab(pd.Series(flat_true_filtered, name='Actual'),
                     pd.Series(flat_pred_filtered, name='Predicted'))

    cm.to_csv(f'outputs/results/confusion_matrix_{index_type}_top{top_n}_filtered.csv')

    # heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, cmap="Blues", annot=True, fmt='d', linewidths=0.5, linecolor='gray')
    plt.title("Top-20 Multi-label Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'outputs/results/confusion_matrix_{index_type}_top{top_n}_filtered.png')
    plt.close()

    print(f" Confusion matrix saved to CSV and PNG (filtered top 20 tags).")

if __name__ == "__main__":
    evaluate_multilabel(index_type='flat', top_n=2)
