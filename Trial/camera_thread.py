#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# Camera Thread Module
# This module handles camera input and frame processing using the ANPR backend
# in a separate thread to keep the GUI responsive.
# ============================================================================

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import time
from typing import Optional, Union, Dict
import logging
from pathlib import Path

class CameraThread(QThread):
    """Thread for handling camera capture operations."""
    
    # Signals
    frame_ready = Signal(np.ndarray)
    error = Signal(str)
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        fps: float = 30.0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ):
        """Initialize camera thread.
        
        Args:
            source: Camera source (device index or video file path)
            fps: Target frames per second
            frame_width: Target frame width (None for default)
            frame_height: Target frame height (None for default)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Camera settings
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_delay = 1.0 / fps
        
        self.running = False
        self.frame = None
        self._capture = None
        
        # Frame processing
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.skip_frames = 2  # Process every nth frame
        
        # ANPR settings
        self.anpr_backend = None
        self.roi_points = None
        self.draw_detections = True
    
    def set_anpr_backend(self, backend):
        """Set the ANPR backend processor."""
        self.anpr_backend = backend
        
    def set_roi(self, points):
        """Set the region of interest for detection."""
        self.roi_points = points
    
    def run(self):
        """Thread main loop."""
        try:
            # Open video capture
            self._capture = cv2.VideoCapture(self.source)
            
            if not self._capture.isOpened():
                raise RuntimeError(f"Failed to open camera source: {self.source}")
            
            # Set frame size if specified
            if self.frame_width is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            if self.frame_height is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            self.running = True
            last_frame_time = 0
            
            while self.running:
                # Control FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                if elapsed < self.frame_delay:
                    continue
                
                # Read frame
                ret, frame = self._capture.read()
                
                if not ret or frame is None:
                    self.error.emit("Failed to read frame from camera")
                    continue
                
                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30 / (current_time - self.last_fps_time)
                    self.last_fps_time = current_time
                
                # Process frame with ANPR backend
                if self.anpr_backend and self.frame_count % self.skip_frames == 0:
                    try:
                        # Run ANPR detection
                        results = self.anpr_backend.process_frame(frame, self.roi_points)
                        
                        # Draw detections on frame
                        frame = self._draw_detections(frame, results)
                        
                    except Exception as e:
                        self.logger.error(f"ANPR processing error: {str(e)}")
                
                # Emit frame
                self.frame = frame
                self.frame_ready.emit(frame)
                last_frame_time = current_time
            
        except Exception as e:
            self.logger.error(f"Camera thread error: {str(e)}")
            self.error.emit(str(e))
        
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop the camera thread."""
        self.running = False
        self.wait()
    
    def _cleanup(self):
        """Clean up resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.frame = None
        self.running = False
    
    def _draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on the frame."""
        try:
            # Draw ROI if set
            if self.roi_points is not None:
                cv2.polylines(frame, [np.array(self.roi_points)], True, (0, 255, 0), 2)
            
            # Draw vehicle detections
            for vehicle in results.get('vehicles', []):
                # Get bounding box
                bbox = vehicle['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw vehicle box
                color = (0, 255, 0)  # Green for vehicles
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw vehicle info
                info_text = f"{vehicle['type']} ({vehicle['color']}) - {vehicle['confidence']:.2f}"
                cv2.putText(frame, info_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw plate detections
                for plate in vehicle.get('plates', []):
                    # Get plate bbox relative to vehicle crop
                    px1, py1, px2, py2 = map(int, plate['bbox'])
                    px1, py1 = px1 + x1, py1 + y1
                    px2, py2 = px2 + x1, py2 + y1
                    
                    # Draw plate box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    
                    # Draw plate text
                    if plate.get('text'):
                        text = f"{plate['text']} ({plate['ocr_confidence']:.2f})"
                        cv2.putText(frame, text, (px1, py2+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing detections: {str(e)}")
            return frame 