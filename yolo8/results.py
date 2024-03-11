from ultralytics import YOLO

# Load a model
model = YOLO('./detect_100_epoch_big/train/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['./datasets/mydatabig/test23/images'], conf=0.35, save=True)  # return a list of Results objects

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
    # result.save()  # save to disk