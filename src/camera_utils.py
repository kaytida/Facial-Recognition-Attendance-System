import cv2
import time

def force_release_camera():
    """
    Force release any existing camera connections.
    This function attempts to release camera resources multiple times to ensure
    the camera is available for the next use.
    """
    print("🔄 Attempting to release any existing camera connections...")
    
    # Try multiple times to release camera
    for i in range(3):
        try:
            temp_cap = cv2.VideoCapture(0)
            if temp_cap.isOpened():
                temp_cap.release()
                print(f"🔓 Released camera connection (attempt {i+1})")
            time.sleep(0.5)
        except:
            pass
    
    # Additional cleanup time
    time.sleep(2.0)

def create_camera_connection():
    """
    Create a new camera connection with retry logic.
    
    Returns:
        cv2.VideoCapture: Camera object if successful, None if failed
    """
    # Try to create new camera connection with retry logic
    video_capture = None
    for attempt in range(3):
        try:
            video_capture = cv2.VideoCapture(0)
            if video_capture.isOpened():
                print("✅ Camera connection established")
                return video_capture
            else:
                video_capture.release()
                time.sleep(1.0)
        except Exception as e:
            print(f"⚠️ Camera connection attempt {attempt+1} failed: {e}")
            if video_capture:
                video_capture.release()
            time.sleep(1.0)
    
    print("❌ Error: Could not access the webcam after multiple attempts.")
    return None

def cleanup_camera_resources(video_capture):
    """
    Comprehensive cleanup of camera resources and OpenCV windows.
    
    Args:
        video_capture: cv2.VideoCapture object to release
    """
    print("🧹 Starting cleanup process...")
    
    # Multiple attempts to release camera
    for i in range(3):
        try:
            if video_capture is not None:
                video_capture.release()
                print(f"🔒 Camera released (attempt {i+1})")
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Warning during camera release attempt {i+1}: {e}")
    
    # Force close all OpenCV windows
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)  # Double wait to ensure cleanup
        print("🪟 All windows closed")
    except Exception as e:
        print(f"⚠️ Warning during window cleanup: {e}")
    
    # Additional cleanup - try to release camera again
    try:
        temp_cleanup = cv2.VideoCapture(0)
        if temp_cleanup.isOpened():
            temp_cleanup.release()
            print("🔓 Final camera cleanup completed")
    except:
        pass
    
    # Give extra time for system cleanup
    time.sleep(3.0)
    print("✅ Cleanup completed - Camera should be available for next use")
