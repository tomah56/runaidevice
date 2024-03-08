from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train14/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['./datasets/mydata/label_valid_yolo/images/18d31b11-IMG_20231108_125011.jpg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk