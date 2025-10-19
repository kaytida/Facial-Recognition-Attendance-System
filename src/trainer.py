from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_classifier_svc(embeddings, labels, model_path="models/face_recognition_model.pkl"):
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)
    joblib.dump(model, model_path)
    print(f"[INFO] Model trained and saved to {model_path}")
    return model


def train_classifier_knn(embeddings, labels, model_path="models/face_recognition_model.pkl", n_neighbors=3, threshold=0.6):
    """
    Train a KNN classifier on embeddings and save it.
    threshold: distance threshold for unknown face detection
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    model.fit(embeddings, labels)
    
    # Save both model and threshold
    joblib.dump({"model": model, "threshold": threshold}, model_path)
    print(f"[INFO] KNN model trained and saved to {model_path} with threshold={threshold}")
    return model, threshold