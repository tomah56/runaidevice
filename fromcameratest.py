from PIL import Image
import pytesseract
import numpy as np
from pytesseract import Output
import cv2
import time

# Initialize the USB camera
camera = cv2.VideoCapture(1)  # Use 0 for the default camera, adjust if needed
# use 1 on debian ARMv12

while True:
    # Capture an image from the camera
    ret, frame = camera.read()

    # Check if the capture was successful
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Convert the OpenCV image to a NumPy array
    img1 = np.array(frame)

    # Perform OCR on the captured image
    text_from_image = pytesseract.image_to_string(img1)

    # Process the OCR results
    results = pytesseract.image_to_data(frame, output_type=Output.DICT)

    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        result_text = results["text"][i]
        conf = int(results["conf"][i])

        if conf > 70:
            result_text = "".join([c if ord(c) < 128 else "" for c in result_text]).strip()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    # Display the result
    cv2.imshow('Result', frame)
    cv2.waitKey(0)

    # Release the camera capture
    camera.release()

    # Wait for 30 seconds before capturing the next image
    time.sleep(30)
