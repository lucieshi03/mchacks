import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

blink_counter = 0
last_blink_time = None
BLINK_TIMEOUT = 4  # seconds
timeout_printed = False

def get_false():
    return False

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
        global blink_counter, last_blink_time, timeout_printed
        current_time = time.time()
        if last_blink_time is None or (current_time - last_blink_time) <= BLINK_TIMEOUT:
            blink_counter += 1
            last_blink_time = current_time
            timeout_printed = False
            return True
    return get_false()

def check_blink_timeout():
    global last_blink_time, timeout_printed
    current_time = time.time()
    if last_blink_time is not None and (current_time - last_blink_time) > BLINK_TIMEOUT:
        if not timeout_printed:
            print(False)
            timeout_printed = True
        last_blink_time = current_time  
        return False
    return True

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_LEFT_EYE,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # Define eye indices
        left_eye_indices = [33, 159, 145, 133, 153, 144]
        right_eye_indices = [362, 386, 374, 263, 380, 373]

        # Detect blink
        blink_detected = detect_wink(face_landmarks.landmark, left_eye_indices, right_eye_indices)
        if blink_detected:
            print(True)
        elif not check_blink_timeout():
            pass

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()