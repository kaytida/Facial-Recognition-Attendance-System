from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings(embeddings, labels, save_path="embeddings_distribution.png"):
    tsne = TSNE(n_components=2, random_state=42,perplexity=min(5, embeddings.shape[0] - 1))
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label)
    plt.legend()
    plt.title("t-SNE visualization of face embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Embedding visualization saved at {save_path}")
