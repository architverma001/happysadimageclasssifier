import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
emotion_model = tf.keras.models.load_model('imgclassifer.h5')  # Replace with your model file name

# Open the camera stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Preprocess the frame
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0  # Normalize pixel values
    
    # Pass the frame through the model to get the prediction
    prediction = emotion_model.predict(np.expand_dims(frame, axis=0))
    
    # Interpret the prediction
    emotion = "Sad" if prediction > 0.70 else "Happy"
    
    # Display the emotion prediction on the frame
    cv2.putText(frame, emotion, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame with the prediction
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
