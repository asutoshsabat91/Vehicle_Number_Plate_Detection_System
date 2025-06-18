import cv2
import numpy as np
import easyocr
import re
import os
import time
from typing import Dict, List, Tuple, Optional

class ANPRModel:
    def __init__(self):
        """
        Initialize the ANPR model with EasyOCR for text recognition.
        """
        # Initialize EasyOCR with English language
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Define preprocessing parameters
        self.resize_width = 640
        self.resize_height = 480
        
        # Common OCR corrections
        self.corrections = {
            'O': '0',  # Letter O to number 0
            'I': '1',  # Letter I to number 1
            'Z': '2',  # Letter Z to number 2
            'S': '5',  # Letter S to number 5
            'B': '8',  # Letter B to number 8
            'D': '0',  # Letter D to number 0
            'Q': '0',  # Letter Q to number 0
            'T': '7',  # Letter T to number 7
        }
        
        # Indian license plate pattern
        # Format: [2 letters][2 digits][2 letters/digits][4 digits]
        # Examples: MH02BH1234, AP07BP3220, KA01AB1234
        self.plate_pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}[A-Z]?$'
        
    def preprocess_image(self, image: np.ndarray, resize_width: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image for better OCR results.
        
        Args:
            image: Input image
            resize_width: Optional width to resize the image to
            
        Returns:
            Tuple of (preprocessed image, resized original image)
        """
        # Use provided resize width or default
        width = resize_width if resize_width else self.resize_width
        
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        aspect_ratio = h / w
        new_width = width
        new_height = int(aspect_ratio * new_width)
        img_resized = cv2.resize(image, (new_width, new_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding with different parameters
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Try CLAHE (Contrast Limited Adaptive Histogram Equalization) on a copy
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(filtered)
        
        # Combine the results
        combined = cv2.bitwise_or(morph, clahe_img)
        
        return combined, img_resized
    
    def recognize_plate(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize license plate text from the image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (recognized text, confidence)
        """
        # Try different preprocessing approaches
        processed_img, resized = self.preprocess_image(image)
        
        # Use EasyOCR to recognize text on processed image
        results_processed = self.reader.readtext(processed_img)
        
        # Also try with the original resized image
        results_original = self.reader.readtext(resized)
        
        # Try with grayscale version of original
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        results_gray = self.reader.readtext(gray)
        
        # Combine all results
        all_results = results_processed + results_original + results_gray
        
        if not all_results:
            return "", 0.0
        
        # Process all results and find the best match
        candidates = []
        for bbox, text, conf in all_results:
            processed_text = self._post_process_text(text)
            if processed_text:  # Only consider non-empty processed text
                candidates.append((processed_text, conf))
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best candidate
        if candidates:
            best_text, best_conf = candidates[0]
            return best_text, best_conf
        
        return "", 0.0
    
    def _post_process_text(self, text: str) -> str:
        """
        Post-process the recognized text to correct common OCR errors.
        
        Args:
            text: Recognized text
            
        Returns:
            Processed text that matches Indian license plate format, or empty string if invalid
        """
        if not text:
            return ""
            
        # Remove spaces and special characters
        text = ''.join(c for c in text if c.isalnum())
        
        # Convert to uppercase
        text = text.upper()
        
        # Skip if too short
        if len(text) < 4:
            return ""
        
        # Apply corrections based on position
        processed_text = ""
        for i, char in enumerate(text):
            # For Indian license plates: first 2 chars are state code (letters),
            # next 1-2 chars are region code (digits),
            # next 1-2 chars are series code (letters),
            # last 4 chars are unique identifier (digits)
            
            # First 2 positions should be letters (state code)
            if i < 2:
                if char.isdigit():
                    # If digit in state code position, try to convert to letter
                    digit_to_letter = {
                        '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'
                    }
                    processed_text += digit_to_letter.get(char, char)
                else:
                    processed_text += char
            # Next 1-2 positions should be digits (region code)
            elif i < 4:
                if char.isalpha() and char in self.corrections:
                    processed_text += self.corrections[char]
                else:
                    processed_text += char
            # Next 1-2 positions should be letters (series code)
            elif i < 6:
                if char.isdigit() and len(text) > 8:  # Only convert if the plate is long enough
                    # If digit in series code position, try to convert to letter
                    digit_to_letter = {
                        '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'
                    }
                    processed_text += digit_to_letter.get(char, char)
                else:
                    processed_text += char
            # Remaining positions should be digits (unique identifier)
            else:
                if char.isalpha() and char in self.corrections:
                    processed_text += self.corrections[char]
                else:
                    processed_text += char
        
        # Check if the text matches Indian license plate pattern
        if re.match(self.plate_pattern, processed_text):
            return processed_text
        
        # If not matching the pattern but looks like a plate (has both letters and numbers)
        # and is of reasonable length, return it anyway
        if (len(processed_text) >= 6 and 
            any(c.isdigit() for c in processed_text) and 
            any(c.isalpha() for c in processed_text)):
            return processed_text
        
        return ""
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process an image to recognize the license plate.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing recognition results including plate text, confidence,
            processing time, and success status
        """
        if image is None:
            return {
                "plate_text": "", 
                "confidence": 0.0, 
                "processing_time": 0.0,
                "success": False
            }
        
        # Start timing
        start_time = time.time()
        
        # Recognize plate text
        plate_text, confidence = self.recognize_plate(image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create a copy of the image for visualization
        result_image = image.copy()
        
        # Draw the result on the image if text was found
        if plate_text:
            # Get image dimensions
            h, w = result_image.shape[:2]
            
            # Draw a semi-transparent background for the text
            overlay = result_image.copy()
            cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
            
            # Draw the text
            cv2.putText(result_image, f"Plate: {plate_text}", (10, h-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_image, f"Conf: {confidence:.2f}", (10, h-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return {
            "plate_text": plate_text,
            "confidence": confidence,
            "processing_time": processing_time,
            "result_image": result_image,
            "success": bool(plate_text)
        }