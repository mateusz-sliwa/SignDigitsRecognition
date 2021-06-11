import tensorflow
import numpy as np
import cv2
import time
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Capturing webcam frames ---------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, [224, 224], fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    # Set webcam frame as an image for recognition
    image = frame
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    if (prediction[0,0] >= 0.7):
        print("0")
    elif (prediction[0,1] >= 0.7):
        print("1")
    elif (prediction[0,2] >= 0.7):
        print("2")
    elif (prediction[0,3] >= 0.7):
        print("3")
    elif (prediction[0,4] >= 0.7):
        print("4")
    elif (prediction[0,5] >= 0.7):
        print("5")
    elif (prediction[0,6] >= 0.7):
        print("6")
    elif (prediction[0,7] >= 0.7):
        print("7")
    elif (prediction[0,8] >= 0.7):
        print("8")
    elif (prediction[0,9] >= 0.7):
        print("9")

    # print(prediction)

    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()