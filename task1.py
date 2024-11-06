import cv2
import os
import time
from picamera2 import Picamera2, Preview

# Initialize variables
num_train_images = 50
num_test_images = 10
img_size = (64, 64)  # Desired size for face images
delay_between_people = 10  # Delay in seconds to allow new person to step in

# Path setup
data_dir = '/home/pi/ee347/lab8/data'
folders = {
    'train': [os.path.join(data_dir, 'train', '0'), os.path.join(data_dir, 'train', '1')],
    'test': [os.path.join(data_dir, 'test', '0'), os.path.join(data_dir, 'test', '1')]
}

# Create directories if they do not exist
for folder in folders['train'] + folders['test']:
    os.makedirs(folder, exist_ok=True)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # Adjust resolution as needed
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Capture and save images for each teammate
for label in [0, 1]:  # 0 for teammate 1, 1 for teammate 2
    train_count = 0
    test_count = 0

    print(f"Starting capture for teammate {label}. Please position yourself in front of the camera.")

    while train_count < num_train_images or test_count < num_test_images:
        # Capture frame-by-frame from Pi Camera
        frame = picam2.capture_array()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
       
        # Process detected faces
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]  # Crop to the face
            face_resized = cv2.resize(face, img_size)  # Resize to 64x64

            # Determine folder path based on image count
            if train_count < num_train_images:
                save_path = os.path.join(folders['train'][label], f"{train_count}.jpg")
                train_count += 1
            elif test_count < num_test_images:
                save_path = os.path.join(folders['test'][label], f"{test_count}.jpg")
                test_count += 1
            else:
                continue  # Exit inner loop if both train and test counts are met

            # Save the processed image
            cv2.imwrite(save_path, face_resized)
            print(f"Saved image to {save_path}")

            # Stop capturing additional faces once required count is met
            if train_count >= num_train_images and test_count >= num_test_images:
                break

        # Display the frame with detected faces for feedback
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Face Capture", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Delay before capturing for the next teammate
    if label == 0:
        print(f"Captured images for teammate {label}. Please wait {delay_between_people} seconds for the next person to step in.")
        time.sleep(delay_between_people)  # Wait before moving to the next person

# Release resources
picam2.stop()
cv2.destroyAllWindows()