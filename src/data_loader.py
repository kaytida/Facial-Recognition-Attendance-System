import os
from glob import glob

def load_dataset(dataset_path="dataset"):
    image_paths, labels = [], []
    for person in os.listdir(dataset_path):  # loop through each folder (e.g., Aditya/)
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        for img_path in glob(os.path.join(person_dir, "*.jpg")):
            image_paths.append(img_path)
            labels.append(person)
    return image_paths, labels
