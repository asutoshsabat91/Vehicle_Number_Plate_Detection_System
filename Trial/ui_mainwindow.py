#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from camera_thread import CameraThread
from anpr_processor import ANPRBackend

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANPR System")
        self.setMinimumSize(800, 600)
        
        # Initialize backend
        self.anpr_backend = ANPRBackend()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create camera view
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.camera_label)
        
        # Create control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setEnabled(False)
        button_layout.addWidget(self.capture_button)
        
        layout.addLayout(button_layout)
        
        # Create result labels
        self.plate_label = QLabel("License Plate: ")
        layout.addWidget(self.plate_label)
        
        self.vehicle_label = QLabel("Vehicle Type: ")
        layout.addWidget(self.vehicle_label)
        
        # Initialize camera thread
        self.camera_thread = None
        self.frame = None
        self.last_detection = None
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33 fps
    
    def start_camera(self):
        if not self.camera_thread:
            self.camera_thread = CameraThread()
            self.camera_thread.set_anpr_backend(self.anpr_backend)
            self.camera_thread.frame_ready.connect(self.on_frame_ready)
            self.camera_thread.error.connect(self.on_camera_error)
            self.camera_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.capture_button.setEnabled(True)
    
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.capture_button.setEnabled(False)
            self.camera_label.clear()
    
    @Slot(np.ndarray)
    def on_frame_ready(self, frame):
        self.frame = frame.copy()
        # Optionally, run detection on every frame for live results
        results, _ = self.anpr_backend.process_frame(self.frame)
        self.last_detection = results
        self.update_labels_from_results(results)
    
    @Slot(str)
    def on_camera_error(self, msg):
        self.camera_label.setText(f"Camera Error: {msg}")
    
    def update_frame(self):
        if self.frame is not None:
            height, width = self.frame.shape[:2]
            bytes_per_line = 3 * width
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # Create QImage from frame
            q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                self.camera_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.camera_label.setPixmap(scaled_pixmap)
    
    def capture_frame(self):
        if self.frame is not None:
            # Process frame with ANPR backend
            results, _ = self.anpr_backend.process_frame(self.frame)
            self.update_labels_from_results(results)
    
    def update_labels_from_results(self, results):
        # Show the first detected plate and vehicle type
        plate_text = ""
        plate_conf = 0.0
        vehicle_type = ""
        if results['vehicles']:
            vehicle = results['vehicles'][0]
            vehicle_type = vehicle['type']
            if vehicle['plates']:
                plate = vehicle['plates'][0]
                plate_text = plate['text']
                plate_conf = plate['ocr_confidence']
        self.plate_label.setText(f"License Plate: {plate_text} (Confidence: {plate_conf:.2f})")
        self.vehicle_label.setText(f"Vehicle Type: {vehicle_type}")
    
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()