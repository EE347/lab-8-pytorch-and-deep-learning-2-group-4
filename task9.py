import os
import time
import torch
import cv2
import torch.nn.functional as F
from picamera2 import Picamera2
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# Function to create a directory to save videos and images
def create_save_directories():
    if not os.path.exists('/home/pi/ee347/lab8/captures/videos'):
        os.makedirs('/home/pi/ee347/lab8/captures/videos')
    if not os.path.exists('/home/pi/ee347/lab8/captures/images'):
        os.makedirs('/home/pi/ee347/lab8/captures/images')

# Function to generate unique filenames
def generate_filename(file_type='image'):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if file_type == 'image':
        return f'/home/pi/ee347/lab8/captures/images/image_{timestamp}.jpg'
    elif file_type == 'video':
        return f'/home/pi/ee347/lab8/captures/videos/video_{timestamp}.avi'

# Initialize Picamera2 for the Raspberry Pi Camera Module 2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())  # Configure for still images initially
picam2.start()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create directories for saving captures
create_save_directories()

# Video Writer for recording video
is_recording = False
video_writer = None
fps = 20.0  # Frames per second for video
frame_width = 640  # Frame width for video (default)
frame_height = 480  # Frame height for video (default)

# Load the trained model for prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
model.load_state_dict(torch.load('lab8/best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the necessary image transforms (e.g., normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((64, 64)),  # Resize to the same size the model was trained on
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Start the camera stream
while True:
    # Capture a frame using picamera2
    frame = picam2.capture_array()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the image

    # Draw a bounding box around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box around the face

        # Crop the face from the image
        cropped_face = frame[y:y + h, x:x + w]

        # Save the cropped face or process it for prediction
        image_filename = generate_filename('image')
        cv2.imwrite(image_filename, cropped_face)
        print(f"Face captured and cropped: {image_filename}")

        # Make prediction using the model on the cropped face
        cropped_face_resized = cv2.resize(cropped_face, (64, 64))
        image_tensor = transform(cropped_face_resized).unsqueeze(0).to(device)  # Add batch dimension and send to device
        with torch.no_grad():
            output = model(image_tensor)  # Forward pass
            _, predicted_class = torch.max(output, 1)  # Get the predicted class
            print(f"Predicted class: {predicted_class.item()}")  # Print the predicted class

    # Resize the frame for video recording (optional)
    frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # Show the camera stream in a window
    cv2.imshow('Camera Stream', frame_resized)

    # Draw a red dot in the top-left corner if recording
    if is_recording:
        cv2.circle(frame_resized, (30, 30), 10, (0, 0, 255), -1)  # Red circle (red dot)

    # Press 'r' to start/stop video recording
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        if not is_recording:
            # Start recording
            video_filename = generate_filename('video')
            video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
            print(f"Recording started: {video_filename}")
        else:
            # Stop recording
            video_writer.release()
            video_writer = None
            print("Recording stopped.")
        is_recording = not is_recording

    # Press 'c' to capture an image
    if key == ord('c'):
        image_filename = generate_filename('image')
        cv2.imwrite(image_filename, frame_resized)
        print(f"Image captured: {image_filename}")

    # Press 'q' to quit the program
    if key == ord('q'):
        break

    # Write the frame to video if recording
    if is_recording:
        video_writer.write(frame_resized)

# Release resources and close all windows
picam2.stop()
cv2.destroyAllWindows()