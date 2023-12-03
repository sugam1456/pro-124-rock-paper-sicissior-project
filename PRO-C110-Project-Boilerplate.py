import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained TensorFlow model
model = tf.keras.models.load_model('your_model_path')  # Replace with your model's path

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Resize the frame to match the input size expected by your model
        resized_frame = cv2.resize(frame, (224, 224))  
       
        expanded_frame = np.expand_dims(resized_frame, axis=0)
        
        normalized_frame = expanded_frame / 255.0  # Normalize to [0, 1] assuming 8-bit color depth
        
        # Get predictions from the model
        predictions = model.predict(normalized_frame)
        
        # Display the predictions (you may need to customize this part)
        prediction_label = np.argmax(predictions, axis=1)
        cv2.putText(frame, f"Prediction: {prediction_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # displaying the frames captured
        cv2.imshow('feed', frame)

        # waiting for 1ms
        code = cv2.waitKey(1)
        
        # if space key is pressed, break the loop
        if code == 32:
            break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
