import cv2
import numpy as np
import time
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

good_counter = 0
bad_counter = 0

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
last_reset_time = time.time()
predicted_label = None  # Initialize the predicted_label variable

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    current_time = time.time()
    if current_time - last_prediction_time >= 1:  # Make predictions every 1 second
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
        # Export counters to a JSON file before resetting
        counters_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Add timestamp
            "good_counter": good_counter,
            "bad_counter": bad_counter
        }

        with open("posture_counters.json", "a") as json_file:  # Append data to the file
            json.dump(counters_data, json_file)
            json_file.write("\n")  # Ensure each record is on a new line

        print(f"Counters reset at {counters_data['timestamp']} - Good: {good_counter}, Bad: {bad_counter}")

        # Reset the counters
        good_counter = 0
        bad_counter = 0
        last_reset_time = current_time

    if predicted_label is not None:
        posture_text = labels.get(predicted_label, "Unknown")
        cv2.putText(frame, f"Posture: {posture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Posture Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# for popups
def isBadPosture(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return True if prediction[0] > 0.5 else False

# for messages
counters_data = {
    "good_counter": good_counter,
    "bad_counter": bad_counter
}

with open("posture_counters.json", "w") as json_file:
    json.dump(counters_data, json_file)

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
