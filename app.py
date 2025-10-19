import streamlit as st
import subprocess
import os
import time
from src.data_loader import load_dataset
from src.embedder import generate_embeddings
from src.trainer import train_classifier_knn
from src.recognizer import recognize_from_webcam_knn
from dataset_collector import collect_images
from src.utils import visualize_embeddings


# -------------------------------
# Callback Functions
# -------------------------------
def update_capture_progress(message, progress_text):
    """Update capture progress display in Streamlit UI."""
    st.session_state.capture_progress.append(message)
    progress_text.text("\n".join(st.session_state.capture_progress[-5:]))  # Show last 5 messages

def update_training_progress(message, training_text):
    """Update training progress display in Streamlit UI."""
    st.session_state.training_progress = getattr(st.session_state, 'training_progress', [])
    st.session_state.training_progress.append(message)
    training_text.text("\n".join(st.session_state.training_progress[-3:]))  # Show last 3 messages


# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Face Recognition Attendance", layout="centered")

st.title("🎓 Face Recognition Attendance System")
st.markdown("---")

# Initialize session state for tracking progress
if 'capture_progress' not in st.session_state:
    st.session_state.capture_progress = []
if 'training_info' not in st.session_state:
    st.session_state.training_info = None
if 'attendance_list' not in st.session_state:
    st.session_state.attendance_list = []

# -------------------------------
# Option 1: Add new user data
# -------------------------------
st.header("📸 Step 1: Add New User Data")

# Instructions
st.info("""
**📋 Instructions for capturing photos:**
1. Enter the person's name below
2. Choose how many photos to capture (recommended: 20-30)
3. Click "Capture Images" to start the camera
4. **Press ENTER key** to take each photo
5. **Press 'q' key** in the camera window to stop early
6. Make sure the person's face is clearly visible in the camera
""")

col1, col2 = st.columns([2, 1])
with col1:
    new_user = st.text_input("Enter new user name:")
with col2:
    num_images = st.number_input("Number of images", min_value=5, max_value=50, value=20)

if st.button("Capture Images"):
    if new_user.strip() == "":
        st.warning("Please enter a valid user name.")
    else:
        st.info(f"Starting camera for {new_user}...")
        
        # Create a placeholder for progress updates
        progress_placeholder = st.empty()
        progress_text = st.empty()
        
        # Capture images with progress callback
        captured_count = collect_images(new_user, num_images, 
                                      progress_callback=lambda msg: update_capture_progress(msg, progress_text))
        
        if captured_count > 0:
            st.success(f"✅ Data collected for {new_user}! Captured {captured_count} images.")
        else:
            st.error("❌ Failed to capture images. Please check your camera.")

st.markdown("---")

# -------------------------------
# Option 2: Train the classifier
# -------------------------------
st.header("🧠 Step 2: Train Classifier")

if st.button("Train on Collected Data"):
    try:
        image_paths, labels = load_dataset()
        if len(image_paths) == 0:
            st.warning("No images found in dataset. Please add some user data first.")
        else:
            st.info(f"Found {len(image_paths)} images in dataset...")
            
            # Create progress display for training
            training_progress = st.empty()
            training_text = st.empty()
            
            with st.spinner("Generating embeddings and training model..."):
                # Generate embeddings with progress
                embeddings, labels = generate_embeddings(image_paths, labels, 
                                                       progress_callback=lambda msg: update_training_progress(msg, training_text))
                
                # Visualize embeddings
                visualize_embeddings(embeddings, labels)
                
                # Train classifier and get training info
                model, threshold, training_info = train_classifier_knn(embeddings, labels)
                st.session_state.training_info = training_info
                
                st.success("✅ Model trained and saved!")
                
                # Display training summary
                st.subheader("📊 Training Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", training_info['total_samples'])
                with col2:
                    st.metric("Unique Classes", training_info['unique_classes'])
                with col3:
                    st.metric("Threshold", f"{training_info['threshold']:.2f}")
                
                st.write("**Classes trained:**", ", ".join(training_info['classes']))
                
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")

st.markdown("---")

# -------------------------------
# Option 3: Run live recognition
# -------------------------------
st.header("🧾 Step 3: Run Attendance Check")

# Instructions for attendance
st.info("""
**📋 Instructions for attendance check:**
1. Make sure you have trained the model first (Step 2)
2. Click "Start Attendance" to begin recognition
3. **Press 'q' key** in the camera window to stop and see results
4. The system will automatically recognize faces and track attendance
5. After closing, you'll see the list of people who attended
""")

if st.button("Start Attendance"):
    if not os.path.exists("models/face_recognition_model.pkl"):
        st.error("❌ No trained model found. Please train the classifier first.")
    else:
        st.info("Starting real-time recognition... (Press 'q' to exit camera)")
        
        # Run recognition and capture attendance
        try:
            attendance_list = recognize_from_webcam_knn()
            st.session_state.attendance_list = attendance_list
            
            # Display attendance results
            st.subheader("📋 Attendance Summary")
            if attendance_list:
                st.success(f"✅ {len(attendance_list)} people attended:")
                for person in attendance_list:
                    st.write(f"• {person}")
            else:
                st.warning("No one was recognized during the session.")
                
        except Exception as e:
            st.error(f"❌ Recognition failed: {str(e)}")

# Display previous attendance if available
if st.session_state.attendance_list:
    st.markdown("---")
    st.subheader("📋 Previous Attendance")
    for person in st.session_state.attendance_list:
        st.write(f"• {person}")
