import os
from google.cloud import vision

# Set the environment variable for Google Cloud Vision API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"your_google_cloud_credentials.json"  # Use the path to your JSON credentials file

# Sample mapping from Ancient Tamil to Modern Tamil (expand this as necessary)
ancient_to_modern_tamil = {
    '𑑀': 'அ',
    '𑑁': 'ஆ',
    '𑑂': 'இ',
    '𑑃': 'ஈ',
    '𑑄': 'உ',
    '𑑅': 'ஊ',
    '𑑆': 'எ',
    '𑑇': 'ஏ',
    '𑑈': 'ஐ',
    '𑑉': 'ஒ',
    '𑑊': 'ஓ',
    '𑑋': 'ஔ',
    '𑑌': 'க',
    '𑑍': 'ங',
    '𑑎': 'ச',
    '𑑏': 'ஞ',
    # Add more mappings as needed
}

def extract_text_from_image(image_path):
    """Extracts text from an image using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()

    # Load the image into memory
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create Image instance
    image = vision.Image(content=content)

    # Perform text detection on the image
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        extracted_text = texts[0].description
        return extracted_text
    else:
        return "No text detected"

def convert_ancient_tamil_to_modern(ancient_tamil_text):
    """Convert Ancient Tamil text to Modern Tamil based on mapping."""
    modern_tamil_text = ''
    
    for char in ancient_tamil_text:
        if char in ancient_to_modern_tamil:
            modern_tamil_text += ancient_to_modern_tamil[char]
        else:
            modern_tamil_text += char  # Keep the character if no mapping exists
    
    return modern_tamil_text

# Example usage
image_path = 'babu.jpg'  # Replace with the path to your image
extracted_text = extract_text_from_image(image_path)

print("Extracted Ancient Tamil Text:", extracted_text)

# Convert extracted text to Modern Tamil
modern_tamil_text = convert_ancient_tamil_to_modern(extracted_text)
print("Converted Modern Tamil Text:", modern_tamil_text)
