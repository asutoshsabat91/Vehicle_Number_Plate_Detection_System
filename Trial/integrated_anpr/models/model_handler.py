from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
import easyocr
from ..utils.image_processing import extract_plate_region

class ModelHandler:
    def __init__(self):
        """Initialize YOLO and OCR models."""
        # Load YOLOv8 model trained on vehicles
        self.yolo_model = YOLO('yolov8n.pt')
        # Initialize EasyOCR with English language
        self.reader = easyocr.Reader(['en'])
        
    def detect_vehicles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect vehicles in image using YOLO.
        Returns list of bounding boxes (x1, y1, x2, y2).
        """
        results = self.yolo_model(image)
        boxes = []
        
        for result in results:
            for box in result.boxes:
                # Class 2 is car, 3 is motorcycle, 5 is bus, 7 is truck in COCO dataset
                if box.cls in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Add some padding around the detected vehicle
                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image.shape[1], x2 + padding)
                    y2 = min(image.shape[0], y2 + padding)
                    boxes.append((x1, y1, x2, y2))
                    
        return boxes
    
    def recognize_plate(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[str]:
        """
        Recognize text in license plate region.
        Returns recognized text or None if no text found.
        """
        plate_img = extract_plate_region(image, bbox)
        if plate_img is None:
            return None
            
        # Use EasyOCR to recognize text
        results = self.reader.readtext(plate_img)
        if results:
            # Get the text with highest confidence
            best_result = max(results, key=lambda x: x[2])
            text = best_result[1]
            
            # Clean up the text (remove spaces and special characters)
            text = ''.join(c for c in text if c.isalnum())
            
            # Validate license plate format
            if self._is_valid_plate(text):
                return text
        return None
    
    def _is_valid_plate(self, text: str) -> bool:
        """
        Validate if the text matches license plate format.
        """
        if len(text) < 5:  # Minimum length for license plates
            return False
            
        # Check if it contains at least 2 letters and 2 numbers
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        return has_letters and has_numbers
    
    def process_frame(
        self, 
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int, int, int], str]]]:
        """
        Process frame to detect vehicles and recognize license plates.
        Returns processed frame and list of (bbox, text) tuples.
        """
        detections = []
        
        # Detect vehicles
        boxes = self.detect_vehicles(frame)
        
        # Process each detection
        for box in boxes:
            text = self.recognize_plate(frame, box)
            if text:
                detections.append((box, text))
                
        return frame, detections 