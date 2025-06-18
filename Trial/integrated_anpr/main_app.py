import os
import sys
import argparse
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QComboBox, QMessageBox, QFrame, QGridLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.anpr_model import ANPRModel
from utils.dataset_loader import DatasetLoader

class ANPRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indian ANPR System - Direct Recognition")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                min-width: 120px;
            }
        """)
        
        # Initialize model
        self.model = ANPRModel()
        
        # Initialize video capture
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize dataset
        self.dataset = None
        self.current_dataset_index = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title_label = QLabel("ANPR Controls")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #333333;
                margin-bottom: 16px;
            }
        """)
        left_layout.addWidget(title_label)
        
        # Input source selection
        source_label = QLabel("Input Source:")
        self.source_combo = QComboBox()
        self.source_combo.addItems(['Camera', 'Video File', 'Image', 'Dataset'])
        left_layout.addWidget(source_label)
        left_layout.addWidget(self.source_combo)
        
        # Dataset file selection
        dataset_label = QLabel("Dataset File:")
        dataset_layout = QHBoxLayout()
        self.dataset_path_label = QLabel("No file selected")
        self.dataset_path_label.setStyleSheet("font-style: italic; color: #666;")
        self.browse_dataset_btn = QPushButton('Browse')
        self.browse_dataset_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(self.dataset_path_label)
        dataset_layout.addWidget(self.browse_dataset_btn)
        left_layout.addWidget(dataset_label)
        left_layout.addLayout(dataset_layout)
        
        # Buttons
        self.start_btn = QPushButton('Start Detection')
        self.start_btn.clicked.connect(self.start_detection)
        left_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton('Stop Detection')
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        left_layout.addWidget(self.stop_btn)
        
        self.browse_btn = QPushButton('Browse File')
        self.browse_btn.clicked.connect(self.browse_file)
        left_layout.addWidget(self.browse_btn)
        
        # Dataset navigation buttons
        dataset_nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton('Previous')
        self.prev_btn.clicked.connect(self.previous_image)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton('Next')
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        
        dataset_nav_layout.addWidget(self.prev_btn)
        dataset_nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(dataset_nav_layout)
        
        # Detection results
        result_frame = QFrame()
        result_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        result_layout = QGridLayout(result_frame)
        
        result_title = QLabel("Detection Results")
        result_title.setStyleSheet("font-weight: bold;")
        result_layout.addWidget(result_title, 0, 0, 1, 2)
        
        # Detected plate
        plate_label = QLabel("Plate:")
        self.plate_text_label = QLabel('No detection yet')
        result_layout.addWidget(plate_label, 1, 0)
        result_layout.addWidget(self.plate_text_label, 1, 1)
        
        # Ground truth (for dataset mode)
        gt_label = QLabel("Ground Truth:")
        self.gt_text_label = QLabel('N/A')
        result_layout.addWidget(gt_label, 2, 0)
        result_layout.addWidget(self.gt_text_label, 2, 1)
        
        # Confidence
        conf_label = QLabel("Confidence:")
        self.conf_text_label = QLabel('N/A')
        result_layout.addWidget(conf_label, 3, 0)
        result_layout.addWidget(self.conf_text_label, 3, 1)
        
        # Match status (for dataset mode)
        match_label = QLabel("Match:")
        self.match_text_label = QLabel('N/A')
        result_layout.addWidget(match_label, 4, 0)
        result_layout.addWidget(self.match_text_label, 4, 1)
        
        left_layout.addWidget(result_frame)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        # Right panel for video display
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 4px;
            }
        """)
        right_layout.addWidget(self.image_label)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 4)
    
    def browse_dataset(self):
        """Browse for dataset file"""
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Dataset File', '', 
                                               'Text Files (*.txt)')
        if file_path:
            self.dataset_path_label.setText(os.path.basename(file_path))
            self.dataset = DatasetLoader(file_path)
            self.current_dataset_index = 0
            
            # Enable dataset navigation if dataset is loaded
            if self.dataset.get_data():
                self.next_btn.setEnabled(True)
                self.prev_btn.setEnabled(False)  # At the beginning
                
                # Auto-select dataset mode
                self.source_combo.setCurrentText('Dataset')
    
    def start_detection(self):
        """Start detection based on selected source"""
        source = self.source_combo.currentText()
        
        if source == 'Camera':
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.timer.start(30)  # 30ms = ~33fps
                self.update_ui_state(True)
            else:
                QMessageBox.warning(self, 'Warning', 'Could not open camera!')
        
        elif source == 'Video File':
            if not hasattr(self, 'video_path'):
                QMessageBox.warning(self, 'Warning', 'Please select a video file first!')
                return
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.timer.start(30)  # 30ms = ~33fps
                self.update_ui_state(True)
            else:
                QMessageBox.warning(self, 'Warning', 'Could not open video file!')
        
        elif source == 'Image':
            if not hasattr(self, 'image_path'):
                QMessageBox.warning(self, 'Warning', 'Please select an image first!')
                return
            self.process_image(self.image_path)
        
        elif source == 'Dataset':
            if not self.dataset or not self.dataset.get_data():
                QMessageBox.warning(self, 'Warning', 'Please load a dataset first!')
                return
            self.process_dataset_image()
    
    def stop_detection(self):
        """Stop detection and release resources"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.update_ui_state(False)
    
    def update_ui_state(self, is_detecting):
        """Update UI state based on detection status"""
        self.start_btn.setEnabled(not is_detecting)
        self.stop_btn.setEnabled(is_detecting)
        self.source_combo.setEnabled(not is_detecting)
        self.browse_btn.setEnabled(not is_detecting)
        self.browse_dataset_btn.setEnabled(not is_detecting)
    
    def browse_file(self):
        """Browse for video or image file"""
        source = self.source_combo.currentText()
        if source == 'Video File':
            file_path, _ = QFileDialog.getOpenFileName(self, 'Select Video', '', 
                                                   'Video Files (*.mp4 *.avi *.mov)')
            if file_path:
                self.video_path = file_path
        else:  # Image
            file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 
                                                   'Image Files (*.jpg *.jpeg *.png)')
            if file_path:
                self.image_path = file_path
                # Auto-process the image
                self.process_image(file_path)
    
    def update_frame(self):
        """Update frame from video source"""
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            result = self.model.process_image(frame)
            
            # Update results
            self.update_result_labels(result, frame)
        else:
            self.stop_detection()
    
    def process_image(self, image_path):
        """Process a single image"""
        frame = cv2.imread(image_path)
        if frame is not None:
            # Process image
            result = self.model.process_image(frame)
            
            # Update results
            self.update_result_labels(result, frame)
        else:
            QMessageBox.warning(self, 'Warning', f'Could not load image: {image_path}')
    
    def process_dataset_image(self):
        """Process current image from dataset"""
        if not self.dataset or not self.dataset.get_data():
            return
        
        data = self.dataset.get_data()
        if self.current_dataset_index >= len(data):
            return
        
        item = data[self.current_dataset_index]
        img_path = item.get("img_path")
        gt_label = item.get("label")
        
        # Load and process image
        image = self.dataset.load_image(img_path)
        if image is not None:
            # Process image
            result = self.model.process_image(image)
            
            # Update ground truth label
            self.gt_text_label.setText(gt_label)
            
            # Update match status
            is_match = result["plate_text"] == gt_label
            self.match_text_label.setText("Yes" if is_match else "No")
            self.match_text_label.setStyleSheet(
                "color: green; font-weight: bold;" if is_match else "color: red; font-weight: bold;"
            )
            
            # Update other results
            self.update_result_labels(result, image, img_path)
        else:
            QMessageBox.warning(self, 'Warning', f'Could not load image: {img_path}')
    
    def update_result_labels(self, result, frame, title=None):
        """Update result labels and display image"""
        # Update plate text
        self.plate_text_label.setText(result["plate_text"] if result["plate_text"] else "Not detected")
        
        # Update confidence
        self.conf_text_label.setText(f"{result['confidence']:.4f}")
        
        # Convert frame to QImage and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio))
        
        # Set window title if provided
        if title:
            self.setWindowTitle(f"Indian ANPR System - {os.path.basename(title)}")
    
    def previous_image(self):
        """Go to previous image in dataset"""
        if not self.dataset or not self.dataset.get_data():
            return
        
        if self.current_dataset_index > 0:
            self.current_dataset_index -= 1
            self.process_dataset_image()
            
            # Update navigation buttons
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(self.current_dataset_index > 0)
    
    def next_image(self):
        """Go to next image in dataset"""
        if not self.dataset or not self.dataset.get_data():
            return
        
        data = self.dataset.get_data()
        if self.current_dataset_index < len(data) - 1:
            self.current_dataset_index += 1
            self.process_dataset_image()
            
            # Update navigation buttons
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(self.current_dataset_index < len(data) - 1)

def main():
    app = QApplication(sys.argv)
    window = ANPRApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()