import cv2
import dlib
import joblib
import numpy as np

def recognize_from_webcam_svc(model_path="models/face_recognition_model.pkl"):
    # Paths
    FACE_RECOGNITION_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"
    SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"

    # Load models
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_encoder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

    # Load your trained classifier
    model = joblib.load(model_path)

    video_capture = cv2.VideoCapture(0)
    recognized_people = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector(rgb_frame, 1)

        encodings = []
        for face in faces:
            # Get dlib landmarks
            shape = shape_predictor(rgb_frame, face)
            # Compute encoding
            encoding = np.array(face_encoder.compute_face_descriptor(rgb_frame, shape, num_jitters=1))
            encodings.append((face, encoding))

        for face, encoding in encodings:
            preds = model.predict([encoding])
            name = preds[0]

            # Scale face rectangle back to original frame size
            top, right, bottom, left = face.top()*4, face.right()*4, face.bottom()*4, face.left()*4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            recognized_people.add(name)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    print("\nAttendance Summary:")
    for person in recognized_people:
        print(f"- {person}")
    


def recognize_from_webcam_knn(model_path="models/face_recognition_model.pkl", min_face_size=50):
    # Paths to dlib models
    FACE_RECOGNITION_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"
    SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"

    # Load dlib models
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_encoder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

    # Load KNN model + threshold
    data = joblib.load(model_path)
    model = data["model"]
    threshold = data["threshold"]

    video_capture = cv2.VideoCapture(0)
    recognized_people = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector(rgb_frame, 1)

        for face in faces:
            # Skip small faces
            if (face.right() - face.left()) < min_face_size or (face.bottom() - face.top()) < min_face_size:
                continue

            # Get landmarks and embeddings
            shape = shape_predictor(rgb_frame, face)
            encoding = np.array(face_encoder.compute_face_descriptor(rgb_frame, shape, num_jitters=1))

            # Predict closest neighbor
            distances, indices = model.kneighbors([encoding])
            distance = distances[0][0]
            if distance < threshold:
                name = model.predict([encoding])[0]
            else:
                name = "Unknown"

            # Scale rectangle back to original frame size
            top, right, bottom, left = [v * 4 for v in (face.top(), face.right(), face.bottom(), face.left())]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            if name != "Unknown":
                recognized_people.add(name)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    print("\nAttendance Summary:")
    for person in recognized_people:
        print(f"- {person}")
