import cv2
import numpy as np
import torch
from model import CNN  # Change NeuralNet to CNN to match the first snippet

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the trained model
model = CNN().to(device)  # Change NeuralNet to CNN
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_image(img):
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)[1]
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    processed_img = preprocess_image(frame)
    
    # Predict
    with torch.no_grad():
        output = model(processed_img)
        prediction = torch.argmax(output, dim=1).item()
    
    # Display the result
    cv2.putText(frame, f"Predicted: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam MNIST', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

