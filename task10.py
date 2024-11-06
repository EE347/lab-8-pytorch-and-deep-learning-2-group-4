import time
import torch
import cv2
from picamera2 import Picamera2
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# Initialize Picamera2 for the Raspberry Pi Camera Module 2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())  # Configure for still images initially
picam2.start()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# Function to map class index to name
def get_class_name(class_idx):
    if class_idx == 0:
        return "Dillon"
    elif class_idx == 1:
        return "Breandan"
    else:
        return "Unknown"

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

        # Make prediction using the model on the cropped face
        cropped_face_resized = cv2.resize(cropped_face, (64, 64))
        image_tensor = transform(cropped_face_resized).unsqueeze(0).to(device)  # Add batch dimension and send to device
        with torch.no_grad():
            output = model(image_tensor)  # Forward pass
            _, predicted_class = torch.max(output, 1)  # Get the predicted class

            # Get the predicted class name based on the class index
            class_name = get_class_name(predicted_class.item())
            print(f"Predicted class: {class_name}")  # Print the predicted class

            # Overlay the class name on the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, class_name, (x, y - 10), font, 0.9, (0, 255, 0), 2)

    # Resize the frame for better display (optional)
    frame_resized = cv2.resize(frame, (640, 480))

    # Show the camera stream in a window
    cv2.imshow('Camera Stream', frame_resized)

    # Press 'q' to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources and close all windows
picam2.stop()
cv2.destroyAllWindows()