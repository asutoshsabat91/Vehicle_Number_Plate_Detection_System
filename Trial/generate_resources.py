#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PIL import Image, ImageDraw

def create_down_arrow():
    """Create a simple down arrow icon."""
    img = Image.new('RGBA', (12, 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a white down arrow
    points = [(2, 4), (10, 4), (6, 8)]
    draw.polygon(points, fill='white')
    
    img.save('down_arrow.png')

def create_checkmark():
    """Create a simple checkmark icon."""
    img = Image.new('RGBA', (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a white checkmark
    points = [(3, 8), (7, 12), (13, 4)]
    draw.line(points, fill='white', width=2)
    
    img.save('checkmark.png')

def create_models_dir():
    """Create models directory."""
    Path('models').mkdir(exist_ok=True)

def main():
    print("Generating resources...")
    
    # Create icons
    create_down_arrow()
    create_checkmark()
    
    # Create directories
    create_models_dir()
    
    print("Resources generated successfully!")

if __name__ == "__main__":
    main() 