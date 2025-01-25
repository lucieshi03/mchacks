import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model (make sure to use the model you trained earlier)
model = load_model("posture_model.h5")

# Define the labels
labels = {True: "Bad", False: "Good"}

# Function to preprocess the image before prediction
def preprocess_image(image):
    # Resize image to the input shape expected by the CNN
    image_resized = cv2.resize(image, (64, 64))
    image_normalized = image_resized / 255.0  # Normalize the image
    image_array = img_to_array(image_normalized)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Start the webcam
cap = cv2.VideoCapture(0)

last_prediction_time = time.time()
predicted_label = None  # Initialize the predicted_label variable

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    current_time = time.time()
    if current_time - last_prediction_time >= 1:  # Changed to >= to allow exact 5 seconds interval
        processed_frame = preprocess_image(frame)

        prediction = model.predict(processed_frame)
        predicted_label = 1 if prediction[0] > 0.5 else 0  # Using a 0.5 threshold for binary classification

        # Display message if bad posture is detected
        if predicted_label:  # "Bad" posture
            print("Bad Posture! Sit up!")

        last_prediction_time = current_time  # Update the time of last prediction

    # Ensure that predicted_label is valid before using it
    if predicted_label is not None:
        # Show the webcam frame with the posture prediction
        posture_text = labels.get(predicted_label, "Unknown")  # Use 'Unknown' if there's an invalid label
        cv2.putText(frame, f"Posture: {posture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Posture Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def isBadPosture(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return True if prediction[0] > 0.5 else False

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
