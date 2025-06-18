import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QComboBox, QMessageBox, QFrame)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from models.model_handler import ModelHandler
from utils.image_processing import draw_detection

class ANPRMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indian ANPR System")
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
        
        # Initialize model handler
        self.model_handler = ModelHandler()
        
        # Initialize video capture
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
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
        
        # Video source selection
        source_label = QLabel("Input Source:")
        self.source_combo = QComboBox()
        self.source_combo.addItems(['Camera', 'Video File', 'Image'])
        left_layout.addWidget(source_label)
        left_layout.addWidget(self.source_combo)
        
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
        result_layout = QVBoxLayout(result_frame)
        
        result_title = QLabel("Detection Results")
        result_title.setStyleSheet("font-weight: bold;")
        result_layout.addWidget(result_title)
        
        self.result_label = QLabel('No detection yet')
        result_layout.addWidget(self.result_label)
        
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
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 4px;
            }
        """)
        right_layout.addWidget(self.video_label)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 4)
        
    def start_detection(self):
        source = self.source_combo.currentText()
        
        if source == 'Camera':
            self.cap = cv2.VideoCapture(0)
        elif source == 'Video File':
            if not hasattr(self, 'video_path'):
                QMessageBox.warning(self, 'Warning', 'Please select a video file first!')
                return
            self.cap = cv2.VideoCapture(self.video_path)
        else:  # Image
            if not hasattr(self, 'image_path'):
                QMessageBox.warning(self, 'Warning', 'Please select an image first!')
                return
            self.process_image(self.image_path)
            return
            
        if self.cap.isOpened():
            self.timer.start(30)  # 30ms = ~33fps
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.source_combo.setEnabled(False)
            self.browse_btn.setEnabled(False)
            
    def stop_detection(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.browse_btn.setEnabled(True)
        
    def browse_file(self):
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
                
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            processed_frame, detections = self.model_handler.process_frame(frame)
            
            # Update results
            if detections:
                latest_text = detections[-1][1]
                self.result_label.setText(f'Detected: {latest_text}')
                
                # Draw all detections
                for bbox, text in detections:
                    processed_frame = draw_detection(processed_frame, bbox, text)
            
            # Convert frame to QImage and display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio))
        else:
            self.stop_detection()
            
    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is not None:
            # Process image
            processed_frame, detections = self.model_handler.process_frame(frame)
            
            # Update results
            if detections:
                latest_text = detections[-1][1]
                self.result_label.setText(f'Detected: {latest_text}')
                
                # Draw all detections
                for bbox, text in detections:
                    processed_frame = draw_detection(processed_frame, bbox, text)
            
            # Convert frame to QImage and display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio))

def main():
    app = QApplication(sys.argv)
    window = ANPRMainWindow()
    window.show()
    sys.exit(app.exec()) 