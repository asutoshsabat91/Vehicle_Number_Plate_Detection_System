import json
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

class DatasetLoader:
    def __init__(self, label_file_path: str):
        """
        Initialize the dataset loader with the path to the label file.
        
        Args:
            label_file_path: Path to the label file containing image paths and labels
        """
        self.label_file_path = label_file_path
        self.data = self._load_labels()
        
    def _load_labels(self) -> List[Dict[str, str]]:
        """
        Load labels from the label file.
        
        Returns:
            List of dictionaries containing image paths and labels
        """
        data = []
        try:
            with open(self.label_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError:
                            print(f"Error parsing line: {line}")
                            continue
            print(f"Loaded {len(data)} items from {self.label_file_path}")
            return data
        except FileNotFoundError:
            print(f"Label file not found: {self.label_file_path}")
            return []
    
    def get_data(self) -> List[Dict[str, str]]:
        """
        Get the loaded data.
        
        Returns:
            List of dictionaries containing image paths and labels
        """
        return self.data
    
    def load_image(self, img_path: str) -> Optional[np.ndarray]:
        """
        Load an image from the given path.
        
        Args:
            img_path: Path to the image
            
        Returns:
            Loaded image as numpy array or None if loading fails
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                return None
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.data:
            return {"count": 0}
        
        # Count label lengths
        label_lengths = {}
        for item in self.data:
            label = item.get("label", "")
            length = len(label)
            label_lengths[length] = label_lengths.get(length, 0) + 1
        
        # Count label patterns (e.g., 2 letters followed by 4 digits)
        patterns = {}
        for item in self.data:
            label = item.get("label", "")
            pattern = ""
            for char in label:
                if char.isalpha():
                    pattern += "A"
                elif char.isdigit():
                    pattern += "D"
                else:
                    pattern += char
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        return {
            "count": len(self.data),
            "label_lengths": label_lengths,
            "patterns": patterns
        }