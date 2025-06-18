import cv2
import numpy as np
from typing import Tuple, List, Optional

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    Optimized for Indian license plates.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def extract_plate_region(
    image: np.ndarray, 
    bbox: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    """
    Extract and preprocess license plate region from image.
    """
    x1, y1, x2, y2 = bbox
    try:
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            return None
            
        # Resize to a standard size for better OCR
        plate_img = cv2.resize(plate_img, (300, 100))
        
        return preprocess_image(plate_img)
    except Exception:
        return None

def draw_detection(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    text: str,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw bounding box and text on image.
    """
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw text background
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    cv2.rectangle(
        image,
        (x1, y1 - text_size[1] - 10),
        (x1 + text_size[0], y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )
    
    return image 