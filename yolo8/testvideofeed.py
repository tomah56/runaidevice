import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

# Load a model
model = YOLO('./runs/detect/train14/weights/best.pt') 

# Initialize the USB camera
camera_index = 1  # Adjust the camera index based on your system
cap = cv2.VideoCapture(camera_index)




# Set up video capture (adjust resolution as needed)
# cap.set(3, 640)  # Width
# cap.set(4, 480)  # Height

while True:
    _, img = cap.read()  # Read a frame from the webcam

    # Convert BGR to RGB (required by YOLOv8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get YOLOv8 predictions
    results = model.predict(img)

    # Create an annotator to draw bounding boxes
    annotator = Annotator(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    # Get the annotated image
    img = annotator.result()

    # Display the annotated image
    cv2.imshow('YOLO V8 Detection', img)

    # Press 'Space' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
