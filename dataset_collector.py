import cv2
import os
from src.camera_utils import force_release_camera, create_camera_connection, cleanup_camera_resources

def collect_images(person_name, num_images=20, progress_callback=None):
    """
    Captures face images from webcam on pressing Enter and stores them in dataset/{person_name}/
    
    Args:
        person_name (str): Name of the person (folder name under dataset/)
        num_images (int): Number of images to capture
        progress_callback (function): Optional callback to report progress
    """
    dataset_path = os.path.join("dataset", person_name)
    os.makedirs(dataset_path, exist_ok=True)

    cap = None
    try:
        # Force release any existing camera connections
        force_release_camera()
        
        # Create new camera connection
        cap = create_camera_connection()
        if cap is None:
            error_msg = "❌ Error: Could not access the webcam after multiple attempts."
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return 0

        start_msg = f"📸 Starting capture for '{person_name}' ..."
        print(start_msg)
        if progress_callback:
            progress_callback(start_msg)
        
        print("➡ Press [ENTER] to capture an image.")
        print("➡ Press [q] in the camera window to quit.\n")

        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                error_msg = "❌ Failed to grab frame. Exiting."
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg)
                break

            # Display live video feed
            cv2.imshow("Dataset Collector - Press 'q' to Quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                stop_msg = "🛑 Capture stopped manually."
                print(stop_msg)
                if progress_callback:
                    progress_callback(stop_msg)
                break

            # Wait for Enter key (key code 13)
            if key == 13:
                img_path = os.path.join(dataset_path, f"{person_name}_{count+1}.jpg")
                cv2.imwrite(img_path, frame)
                success_msg = f"✅ Saved {img_path} ({count+1}/{num_images})"
                print(success_msg)
                if progress_callback:
                    progress_callback(success_msg)
                count += 1

    except Exception as e:
        error_msg = f"❌ Error during capture: {str(e)}"
        print(error_msg)
        if progress_callback:
            progress_callback(error_msg)
    finally:
        # Comprehensive cleanup of camera resources
        cleanup_camera_resources(cap)
    
    final_msg = f"✅ Done! Captured {count} images for '{person_name}' in '{dataset_path}'."
    print(final_msg)
    if progress_callback:
        progress_callback(final_msg)
    
    return count


if __name__ == "__main__":
    person_name = input("Enter the name of the person: ").strip()
    num_images = input("Enter number of images to capture (default 20): ").strip()
    num_images = int(num_images) if num_images.isdigit() else 20

    collect_images(person_name, num_images=num_images)
