import face_recognition
import numpy as np

def compute_face_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image,model = 'cnn')
    if not face_locations:
        print(f"No face found in {image_path}")
        return None
    encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
    return encoding


def generate_embeddings(image_paths, labels, progress_callback=None):
    embeddings, valid_labels = [], []
    total_images = len(image_paths)
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels), 1):
        progress_msg = f"Processing image {i}/{total_images}: {img_path}"
        print(progress_msg)
        if progress_callback:
            progress_callback(progress_msg)
            
        embedding = compute_face_embedding(img_path)
        if embedding is not None:
            embeddings.append(embedding)
            valid_labels.append(label)
            success_msg = f"✅ Successfully processed: {img_path}"
            print(success_msg)
            if progress_callback:
                progress_callback(success_msg)
        else:
            error_msg = f"❌ Failed to process: {img_path}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
    
    final_msg = f"Embedding generation complete! Processed {len(embeddings)}/{total_images} images successfully."
    print(final_msg)
    if progress_callback:
        progress_callback(final_msg)
    
    return np.array(embeddings), np.array(valid_labels)
