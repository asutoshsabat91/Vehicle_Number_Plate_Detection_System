import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_anpr.utils.dataset_loader import DatasetLoader
from integrated_anpr.models.anpr_model import ANPRModel

def calculate_accuracy(predictions, ground_truth):
    """
    Calculate accuracy metrics for the predictions.
    
    Args:
        predictions: List of predicted license plate texts
        ground_truth: List of ground truth license plate texts
        
    Returns:
        Dictionary containing accuracy metrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    correct = 0
    character_correct = 0
    total_characters = 0
    
    for pred, gt in zip(predictions, ground_truth):
        # Exact match
        if pred == gt:
            correct += 1
        
        # Character-level accuracy
        min_len = min(len(pred), len(gt))
        for i in range(min_len):
            if pred[i] == gt[i]:
                character_correct += 1
        total_characters += max(len(pred), len(gt))
    
    exact_accuracy = correct / len(predictions) if predictions else 0
    character_accuracy = character_correct / total_characters if total_characters else 0
    
    return {
        "exact_accuracy": exact_accuracy,
        "character_accuracy": character_accuracy,
        "correct": correct,
        "total": len(predictions),
        "character_correct": character_correct,
        "total_characters": total_characters
    }

def evaluate_model(label_file_path, output_file=None):
    """
    Evaluate the ANPR model on the dataset.
    
    Args:
        label_file_path: Path to the label file
        output_file: Path to save the evaluation results
        
    Returns:
        Dictionary containing evaluation results
    """
    # Load dataset
    dataset = DatasetLoader(label_file_path)
    data = dataset.get_data()
    
    if not data:
        print("No data loaded from the dataset")
        return {}
    
    # Initialize model
    model = ANPRModel()
    
    # Process each image
    predictions = []
    ground_truth = []
    confidences = []
    failed_images = []
    
    print(f"Evaluating model on {len(data)} images...")
    for item in tqdm(data):
        img_path = item.get("img_path")
        label = item.get("label")
        
        # Load image
        image = dataset.load_image(img_path)
        if image is None:
            failed_images.append(img_path)
            continue
        
        # Process image
        result = model.process_image(image)
        
        # Store results
        predictions.append(result["plate_text"])
        ground_truth.append(label)
        confidences.append(result["confidence"])
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, ground_truth)
    
    # Prepare results
    results = {
        "accuracy": accuracy,
        "failed_images": failed_images,
        "predictions": list(zip(ground_truth, predictions, confidences))
    }
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Exact Match Accuracy: {accuracy['exact_accuracy']:.4f} ({accuracy['correct']}/{accuracy['total']})")
    print(f"Character-level Accuracy: {accuracy['character_accuracy']:.4f} ({accuracy['character_correct']}/{accuracy['total_characters']})")
    print(f"Failed to load {len(failed_images)} images")
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Evaluation Results:\n")
            f.write(f"Exact Match Accuracy: {accuracy['exact_accuracy']:.4f} ({accuracy['correct']}/{accuracy['total']})\n")
            f.write(f"Character-level Accuracy: {accuracy['character_accuracy']:.4f} ({accuracy['character_correct']}/{accuracy['total_characters']})\n")
            f.write(f"Failed to load {len(failed_images)} images\n\n")
            
            f.write(f"Detailed Results:\n")
            f.write(f"{'Ground Truth':<15} {'Prediction':<15} {'Confidence':<10} {'Correct':<10}\n")
            for gt, pred, conf in zip(ground_truth, predictions, confidences):
                f.write(f"{gt:<15} {pred:<15} {conf:<10.4f} {gt == pred}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate ANPR model on dataset')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file')
    parser.add_argument('--output', type=str, help='Path to save the evaluation results')
    args = parser.parse_args()
    
    evaluate_model(args.label_file, args.output)

if __name__ == '__main__':
    main()