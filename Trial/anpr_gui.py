#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSettings, QCoreApplication
from ui_mainwindow import MainWindow

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def setup_qt_environment():
    """Setup Qt environment variables and plugins."""
    logger = logging.getLogger(__name__)
    
    # Get the PySide6 installation directory
    import PySide6
    pyside6_dir = os.path.dirname(PySide6.__file__)
    
    # Set up platform plugin path
    platforms_path = os.path.join(pyside6_dir, 'Qt', 'plugins', 'platforms')
    if os.path.exists(platforms_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platforms_path
        logger.info(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to: {platforms_path}")
    
    # Set DYLD_LIBRARY_PATH to the Qt lib directory
    qt_lib_path = os.path.join(pyside6_dir, 'Qt', 'lib')
    if os.path.exists(qt_lib_path):
        os.environ['DYLD_LIBRARY_PATH'] = qt_lib_path
        logger.info(f"Set DYLD_LIBRARY_PATH to: {qt_lib_path}")
    
    # Set additional environment variables for macOS
    os.environ['QT_QPA_PLATFORM'] = 'cocoa'
    
    # Log environment variables for debugging
    logger.info(f"QT_QPA_PLATFORM_PLUGIN_PATH: {os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', 'Not set')}")
    logger.info(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM', 'Not set')}")
    logger.info(f"DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH', 'Not set')}")
    logger.info(f"DYLD_FRAMEWORK_PATH: {os.environ.get('DYLD_FRAMEWORK_PATH', 'Not set')}")

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Setup Qt environment
        setup_qt_environment()
        
        # Enable high DPI scaling (modern approach)
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi)
        
        app = QApplication(sys.argv)
        
        # Set application info for QSettings
        app.setOrganizationName("ANPR Solutions")
        app.setApplicationName("ANPR GUI")
        
        # Load stylesheet
        style_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.qss")
        if os.path.exists(style_file):
            with open(style_file, "r") as f:
                app.setStyleSheet(f.read())
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()