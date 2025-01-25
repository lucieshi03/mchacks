import cv2
import os
import time

def capture_images(label, num_images=50):
    folder_name = f"{label}_postures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    cap = cv2.VideoCapture(0)

    print(f"Capturing {num_images} {label} postures...")
    captured = 0
    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed.")
            break
            
        cv2.imshow("Capturing Image", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            # Save the captured image with a unique filename
            image_filename = os.path.join(folder_name, f"{label}_{captured+1}.jpg")
            cv2.imwrite(image_filename, frame)
            print(f"Captured {label} posture {captured+1}")
            captured += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured} {label} postures!")

# Example: Capture 50 good postures
capture_images("good", 50)

# Example: Capture 50 bad postures
capture_images("bad", 50)