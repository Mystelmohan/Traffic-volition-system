# import cv2
# import numpy as np
# from ultralytics import YOLO
# from paddleocr import PaddleOCR

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load YOLO model (Replace "mohan.pt" with your trained YOLO model path)
# model = YOLO("mohan.pt")

# # Load image
# image_path = "img.png"  # Path to your input image
# image = cv2.imread(image_path)

# # Check if image is loaded correctly
# if image is None:
#     print("❌ Error: Image not found.")
#     exit()

# # Run YOLO model inference to detect license plates
# results = model(image)

# # Process results
# plate_detected = False
# for result in results:
#     for box in result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#         confidence = box.conf[0].item()  # Confidence score
#         class_id = int(box.cls[0])  # Class ID

#         # ✅ Ensure correct class ID for license plates (assuming class_id 1 is for license plates)
#         if class_id == 1 and confidence > 0.5:  # Adjust confidence threshold if needed
#             plate_detected = True

#             # Crop the detected license plate from the image
#             cropped_plate = image[y1:y2, x1:x2]

#             # Save the cropped plate for verification
#             cv2.imwrite("cropped_plate.jpg", cropped_plate)
#             print(f"✅ License plate cropped and saved as cropped_plate.jpg")

#             # Preprocess the cropped image for OCR
#             gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#             gray = cv2.equalizeHist(gray)  # Histogram equalization for contrast enhancement
#             blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Gaussian blur to reduce noise
#             adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#             # Save the processed image for OCR
#             cv2.imwrite("processed_plate.jpg", adaptive_thresh)

#             # Perform OCR using PaddleOCR
#             ocr_result = ocr.ocr(adaptive_thresh, cls=True)

#             # Extract text from OCR result
#             if ocr_result:
#                 plate_number = ""
#                 for line in ocr_result:
#                     if line:
#                         for word in line:
#                             plate_number += word[1][0] + " "  # Concatenate detected text

#                 plate_number = plate_number.strip()  # Clean up the text
#                 print("🚗 Detected License Plate:", plate_number)
#             else:
#                 print("❌ OCR failed to detect any text.")

#         # Draw bounding box on the image for visualization
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, f"Class {class_id}: {confidence:.2f}",
#                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # If no license plate was detected
# if not plate_detected:
#     print("❌ No license plate detected.")

# # Save the final image with bounding boxes
# cv2.imwrite("YOLO_Detection_Result.jpg", image)

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from inference_sdk import InferenceHTTPClient

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="OqzVg3h0ZTssMowCV2Pg"
)

# Load YOLO model
model = YOLO("mohan.pt")  # Replace with your trained YOLO model

# Load image
image_path = "img.png"
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    print("❌ Error: Image not found.")
    exit()

# Run YOLO detection
results = model(image)

# Process results
plate_detected = False

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0])  # Class ID

        # ✅ Ensure correct class ID for license plates
        if class_id == 1 and confidence > 0.5:  # Adjust confidence if needed
            plate_detected = True

            # Crop detected license plate
            cropped_plate = image[y1:y2, x1:x2]

            # Save cropped plate for debugging
            cv2.imwrite("cropped_plate.jpg", cropped_plate)
            print(f"✅ License plate saved: cropped_plate.jpg")

            # === 🔥 Advanced Image Preprocessing for OCR 🔥 ===
            gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            # 🔹 Histogram Equalization to improve contrast
            gray = cv2.equalizeHist(gray)

            # 🔹 Gaussian Blur (removes noise while keeping edges)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # 🔹 Adaptive Thresholding for better text extraction
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_plate = cv2.filter2D(adaptive_thresh, -1, kernel)

            # Save processed image for debugging
            cv2.imwrite("processed_plate.jpg", sharpened_plate)

            # === 🔥 Send Image to Roboflow API for license plate extraction 🔥 ===
            # Send the cropped image to the Roboflow API for plate extraction
            result = CLIENT.infer("cropped_plate.jpg", model_id="license-plate-character-extraction/2")

            # Check the response from Roboflow (no .json() call needed)
            if 'predictions' in result and len(result['predictions']) > 0:
                # Sort predictions first by the 'y' position (top-to-bottom), then by 'x' position (left-to-right)
                sorted_predictions = sorted(result['predictions'], key=lambda x: (x['y'], x['x']))

                # Initialize variables to hold the groups and the final text result
                grouped_predictions = []
                current_group = []
                prev_y = sorted_predictions[0]['y']

                # Group characters based on their vertical (y) and horizontal (x) positions
                for prediction in sorted_predictions:
                    # If the vertical distance is large enough (indicating different line), we create a new group
                    if abs(prediction['y'] - prev_y) > 20:  # Adjust the threshold for y-distance if necessary
                        grouped_predictions.append(current_group)
                        current_group = []
                    current_group.append(prediction['class'])  # Add the character to the current group
                    prev_y = prediction['y']

                # Append the last group
                if current_group:
                    grouped_predictions.append(current_group)

                # Now combine all groups into a single string
                extracted_text = ''.join([''.join(group) for group in grouped_predictions])

                print("🚗 Detected License Plate from Roboflow:", extracted_text)
            else:
                print("❌ Roboflow failed to detect any license plate text.")

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Class {class_id}: {confidence:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# If no license plate was detected
if not plate_detected:
    print("❌ No license plate detected.")

# Save the final image with bounding boxes
cv2.imwrite("YOLO_Detection_Result.jpg", image)
print("✅ Final detection result saved as YOLO_Detection_Result.jpg")



# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# # Load image (bullet.jpg)
# image_path = "bullet.jpg"
# image = cv2.imread(image_path)

# # Check if image is loaded correctly
# if image is None:
#     print("❌ Error: Image not found.")
#     exit()

# # === 🔥 Advanced Image Preprocessing for OCR 🔥 ===
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# # 🔹 Histogram Equalization to improve contrast
# gray = cv2.equalizeHist(gray)

# # 🔹 Gaussian Blur (removes noise while keeping edges)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# # 🔹 Adaptive Thresholding for better text extraction
# adaptive_thresh = cv2.adaptiveThreshold(
#     blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# sharpened_image = cv2.filter2D(adaptive_thresh, -1, kernel)

# # Save processed image
# cv2.imwrite("processed_bullet.jpg", sharpened_image)

# # === 🔥 OCR using PaddleOCR 🔥 ===
# result = ocr.ocr(sharpened_image, cls=True)

# # Check if OCR result is None or empty
# if result and isinstance(result, list):
#     plate_number = ""
#     for line in result:
#         if line:  # Check if line is not None or empty
#             for word in line:
#                 plate_number += word[1][0] + " "  # Extract detected text

#     plate_number = plate_number.strip().replace("\n", "").replace("\f", "")
#     print("🚗 Detected Text from bullet.jpg:", plate_number)
# else:
#     print("❌ OCR failed to detect any text.")
