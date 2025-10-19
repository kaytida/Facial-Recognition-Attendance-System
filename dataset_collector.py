import cv2
import os

def collect_images(person_name, num_images=20):
    """
    Captures face images from webcam on pressing Enter and stores them in dataset/{person_name}/
    
    Args:
        person_name (str): Name of the person (folder name under dataset/)
        num_images (int): Number of images to capture
    """
    dataset_path = os.path.join("dataset", person_name)
    os.makedirs(dataset_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print(f"📸 Starting capture for '{person_name}' ...")
    print("➡ Press [ENTER] to capture an image.")
    print("➡ Press [q] in the camera window to quit.\n")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame. Exiting.")
            break

        # Display live video feed
        cv2.imshow("Dataset Collector - Press 'q' to Quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("🛑 Capture stopped manually.")
            break

        # Wait for Enter key (key code 13)
        if key == 13:
            img_path = os.path.join(dataset_path, f"{person_name}_{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"✅ Saved {img_path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! Captured {count} images for '{person_name}' in '{dataset_path}'.")


if __name__ == "__main__":
    person_name = input("Enter the name of the person: ").strip()
    num_images = input("Enter number of images to capture (default 20): ").strip()
    num_images = int(num_images) if num_images.isdigit() else 20

    collect_images(person_name, num_images=num_images)
