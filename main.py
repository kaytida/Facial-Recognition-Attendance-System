from src.data_loader import load_dataset
from src.embedder import generate_embeddings
from src.utils import visualize_embeddings
from src.trainer import train_classifier_svc
from src.trainer import train_classifier_knn
from src.recognizer import recognize_from_webcam_svc
from src.recognizer import recognize_from_webcam_knn

def main():
    print("[STEP 1] Loading dataset...")
    image_paths, labels = load_dataset()

    print("[STEP 2] Generating embeddings...")
    embeddings, labels = generate_embeddings(image_paths, labels)

    print("[STEP 3] Visualizing embeddings...")
    visualize_embeddings(embeddings, labels)

    print("[STEP 4] Training classifier...")
    train_classifier_knn(embeddings, labels)

    print("[STEP 5] Starting real-time recognition...")
    recognize_from_webcam_knn()

if __name__ == "__main__":
    main()
