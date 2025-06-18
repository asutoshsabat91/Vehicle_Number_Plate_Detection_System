# ANPR Backend class for vehicle detection and license plate recognition
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
import logging
from typing import Dict, List, Tuple, Optional, Union
import re

class ANPRBackend:
    # Initialize the ANPR backend with all necessary models and configurations
    def __init__(
        self,
        vehicle_model_path: Union[str, Path] = 'models/vehicle_detector.pt',
        plate_model_path: Union[str, Path] = 'models/license_plate_detector.pt',
        classifier_model_path: Union[str, Path] = 'models/vehicle_type_classifier.pt',
        device: str = 'cpu',
        confidence: float = 0.25
    ):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.confidence = confidence
        
        # Load pretrained models only
        try:
            self.vehicle_detector = YOLO(vehicle_model_path)
            self.plate_detector = YOLO(plate_model_path)
            self.vehicle_classifier = YOLO(classifier_model_path)
            self.ocr = easyocr.Reader(['en'], gpu=False)
            self.logger.info("All pretrained models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
        
        # Color ranges in HSV
        self.color_ranges = {
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 30)],
            'red': [(0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (35, 255, 255)],
            'green': [(35, 50, 50), (85, 255, 255)],
            'silver': [(0, 0, 140), (180, 30, 200)],
            'grey': [(0, 0, 70), (180, 30, 140)]
        }
    
    # Detect vehicles in the frame
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        try:
            results = self.vehicle_detector(frame, conf=self.confidence, device=self.device)[0]
            detections = []
            
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                detection = {
                    'bbox': bbox,
                    'confidence': float(box.conf),
                    'class_id': int(box.cls),
                    'class_name': results.names[int(box.cls)]
                }
                detections.append(detection)
            
            return detections
        except Exception as e:
            self.logger.error(f"Vehicle detection error: {str(e)}")
            return []
    
    # Detect license plates in the frame
    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        try:
            results = self.plate_detector(frame, conf=self.confidence, device=self.device)[0]
            detections = []
            
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                detection = {
                    'bbox': bbox,
                    'confidence': float(box.conf)
                }
                detections.append(detection)
            
            return detections
        except Exception as e:
            self.logger.error(f"License plate detection error: {str(e)}")
            return []
    
    # Classify vehicle type from cropped image
    def classify_vehicle(self, vehicle_crop: np.ndarray) -> Tuple[str, float]:
        try:
            results = self.vehicle_classifier(vehicle_crop, device=self.device)[0]
            class_id = results.probs.top1
            confidence = float(results.probs.top1conf)
            class_name = results.names[class_id]
            return class_name, confidence
        except Exception as e:
            self.logger.error(f"Vehicle classification error: {str(e)}")
            return "unknown", 0.0
    
    # Detect the dominant color of the vehicle
    def detect_color(self, vehicle_crop: np.ndarray) -> Tuple[str, float]:
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            
            max_pixels = 0
            detected_color = "unknown"
            
            for color, ranges in self.color_ranges.items():
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                if len(ranges) == 2:  # Single range
                    mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                else:  # Multiple ranges (e.g., for red)
                    for i in range(0, len(ranges), 2):
                        mask = cv2.bitwise_or(
                            mask,
                            cv2.inRange(hsv, np.array(ranges[i]), np.array(ranges[i+1]))
                        )
                
                pixels = cv2.countNonZero(mask)
                if pixels > max_pixels:
                    max_pixels = pixels
                    detected_color = color
            
            confidence = max_pixels / (hsv.shape[0] * hsv.shape[1])
            return detected_color, confidence
        except Exception as e:
            self.logger.error(f"Color detection error: {str(e)}")
            return "unknown", 0.0
    
    # Recognize text on the license plate
    def recognize_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        try:
            # Preprocess the plate image
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR
            result = self.ocr.readtext(enhanced)
            if not result or not result[0]:
                return "", 0.0
            
            # Get the text with highest confidence
            texts_and_confidences = []
            for line in result:
                text = line[1]
                conf = line[2]
                texts_and_confidences.append((text, conf))
            
            if not texts_and_confidences:
                return "", 0.0
            
            # Sort by confidence and get the best result
            text, conf = max(texts_and_confidences, key=lambda x: x[1])
            
            # Post-process the text
            text = self._post_process_plate_text(text)
            
            return text, conf
        except Exception as e:
            self.logger.error(f"Plate recognition error: {str(e)}")
            return "", 0.0
    
    # Post-process the recognized plate text
    def _post_process_plate_text(self, text: str) -> str:
        # Remove spaces and special characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Indian license plate format: AA11AA1111
        # where A is letter and 1 is digit
        if len(text) >= 6:  # Minimum valid length
            # Try to correct common OCR mistakes
            text = text.replace('0', 'O').replace('1', 'I').replace('8', 'B')
            
            # Extract what looks like a valid plate number
            plate_pattern = r'[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}'
            matches = re.findall(plate_pattern, text)
            
            if matches:
                return matches[0]
        
        return text
    
    # Process a frame and return vehicle detections with visualization
    def process_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Dict, np.ndarray]:
        if roi:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        
        # Make a copy for visualization
        viz_frame = frame.copy()
        
        # Detect vehicles
        vehicle_detections = self.detect_vehicles(frame)
        results = {'vehicles': []}
        
        for vehicle in vehicle_detections:
            bbox = vehicle['bbox']
            v_x1, v_y1, v_x2, v_y2 = [int(x) for x in bbox]
            vehicle_crop = frame[v_y1:v_y2, v_x1:v_x2]
            
            # Get vehicle type and color
            vehicle_type, type_conf = self.classify_vehicle(vehicle_crop)
            color, color_conf = self.detect_color(vehicle_crop)
            
            # Detect license plate
            plate_detections = self.detect_plates(vehicle_crop)
            plates = []
            
            if plate_detections:
                # Get the plate with highest confidence
                best_plate = max(plate_detections, key=lambda x: x['confidence'])
                p_bbox = best_plate['bbox']
                p_x1, p_y1, p_x2, p_y2 = [int(x) for x in p_bbox]
                
                # Get plate coordinates relative to full frame
                p_x1, p_x2 = p_x1 + v_x1, p_x2 + v_x1
                p_y1, p_y2 = p_y1 + v_y1, p_y2 + v_y1
                
                # Recognize plate text
                plate_crop = frame[p_y1:p_y2, p_x1:p_x2]
                plate_text, plate_conf = self.recognize_plate(plate_crop)
                
                plates.append({
                    'bbox': [p_x1 - v_x1, p_y1 - v_y1, p_x2 - v_x1, p_y2 - v_y1],
                    'text': plate_text,
                    'ocr_confidence': plate_conf,
                    'confidence': float(best_plate['confidence'])
                })
            
            # Store results
            vehicle_result = {
                'bbox': vehicle['bbox'],
                'type': vehicle_type,
                'color': color,
                'confidence': type_conf,
                'plates': plates
            }
            results['vehicles'].append(vehicle_result)
        
        return results, viz_frame