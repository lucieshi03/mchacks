import cv2
import numpy as np
import time
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

good_counter = 0
bad_counter = 0
blink_bad_counter = 0
last_blink_time = None
BLINK_TIMEOUT = 3  # seconds
timeout_printed = False

model = load_model("posture_model.h5")
labels = {True: "Bad", False: "Good"}

# MediaPipe setup for wink detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def preprocess_image(image):
    image_resized = cv2.resize(image, (64, 64))
    image_normalized = image_resized / 255.0
    image_array = img_to_array(image_normalized)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension???????????????????
    return image_array

def detect_wink(face_landmarks, left_eye_indices, right_eye_indices):
    def eye_aspect_ratio(eye_indices):
        vertical1 = ((face_landmarks[eye_indices[1]].x - face_landmarks[eye_indices[5]].x) ** 2 +
                     (face_landmarks[eye_indices[1]].y - face_landmarks[eye_indices[5]].y) ** 2) ** 0.5
        vertical2 = ((face_landmarks[eye_indices[2]].x - face_landmarks[eye_indices[4]].x) ** 2 +
                     (face_landmarks[eye_indices[2]].y - face_landmarks[eye_indices[4]].y) ** 2) ** 0.5
        horizontal = ((face_landmarks[eye_indices[0]].x - face_landmarks[eye_indices[3]].x) ** 2 +
                      (face_landmarks[eye_indices[0]].y - face_landmarks[eye_indices[3]].y) ** 2) ** 0.5
        return (vertical1 + vertical2) / (2.0 * horizontal)

    EAR_THRESHOLD = 0.2  
    left_ear = eye_aspect_ratio(left_eye_indices)
    right_ear = eye_aspect_ratio(right_eye_indices)

    if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
        global last_blink_time, timeout_printed
        current_time = time.time()
        if last_blink_time is None or (current_time - last_blink_time) <= BLINK_TIMEOUT:
            last_blink_time = current_time
            timeout_printed = False
            return True
    return False

def check_blink_timeout():
    global last_blink_time, timeout_printed, blink_bad_counter
    current_time = time.time()
    if last_blink_time is not None and (current_time - last_blink_time) > BLINK_TIMEOUT:
        if not timeout_printed:
            blink_bad_counter+=1
            print(False)
            timeout_printed = True
        last_blink_time = current_time  
        return False
    return True

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  
    last_prediction_time = time.time()
    last_reset_time = time.time()
    predicted_label = None  

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()

        if current_time - last_prediction_time >= 1:
            processed_frame = preprocess_image(frame)
            prediction = model.predict(processed_frame)
            predicted_label = 1 if prediction[0] > 0.5 else 0

            if predicted_label:  # "Bad" posture
                bad_counter += 1
                print("Bad Posture! Sit up!")
            else:
                good_counter += 1

            last_prediction_time = current_time

        # Reset counters every 60 seconds
        if current_time - last_reset_time >= 60:
            counters_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "good_counter": good_counter,
                "bad_counter": bad_counter,
                "blink_bad_counter": blink_bad_counter
            }
            with open("counters.json", "a") as json_file:
                json.dump(counters_data, json_file)
                json_file.write("\n")
            print(f"Counters reset at {counters_data['timestamp']} - good_posture: {good_counter}, bad_posture: {bad_counter}, blink_bad_counter: {blink_bad_counter}")

            good_counter = 0
            bad_counter = 0
            blink_bad_counter = 0
            last_reset_time = current_time

        # Perform wink detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_indices = [33, 159, 145, 133, 153, 144]
                right_eye_indices = [362, 386, 374, 263, 380, 373]

                blink_detected = detect_wink(face_landmarks.landmark, left_eye_indices, right_eye_indices)
                if blink_detected:
                    print("Wink Detected!")
                elif not check_blink_timeout():
                    pass

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        # Display posture label
        if predicted_label is not None:
            posture_text = labels.get(predicted_label, "Unknown")
            cv2.putText(frame, f"Posture: {posture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Posture and Wink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counters_data = {
        "good_counter": good_counter,
        "bad_counter": bad_counter,
        "blink_bad_counter": blink_bad_counter
    }
    with open("counters.json", "w") as json_file:
        json.dump(counters_data, json_file)

cap.release()
cv2.destroyAllWindows()
