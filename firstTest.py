from PIL import Image
import pytesseract
import numpy as np
from pytesseract import Output
import cv2

# filename = '1_python-ocr.jpg'
# filename = './img/cabodgecutedclear.jpeg'
filename = './img/mustcutedclear.jpeg'
img1 = np.array(Image.open(filename))

text_from_image = pytesseract.image_to_string(img1)

image = cv2.imread(filename)
results = pytesseract.image_to_data(image, output_type=Output.DICT)

for i in range(0, len(results["text"])):
   x = results["left"][i]
   y = results["top"][i]

   w = results["width"][i]
   h = results["height"][i]

   result_text = results["text"][i]
   conf = int(results["conf"][i])

   if conf > 70:
       result_text = "".join([c if ord(c) < 128 else "" for c in result_text]).strip()
       cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cv2.putText(image, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

cv2.imshow('Result', image)
cv2.waitKey(0)

print(text_from_image)
