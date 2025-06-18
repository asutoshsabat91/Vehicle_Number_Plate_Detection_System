import os
import json
import argparse
from pathlib import Path

def create_dataset_file(output_path, data_entries=None):
    """
    Create a dataset file with the provided entries or use default entries.
    
    Args:
        output_path: Path to save the dataset file
        data_entries: List of dictionaries with img_path and label keys
    """
    # Default entries from the user's input if none provided
    if data_entries is None:
        data_entries = [
            {"img_path": "D:/Trial/datasets/val/01CC1A0001.jpg", "label": "01CC1A0001"},
            {"img_path": "D:/Trial/datasets/val/07CY9409.jpg", "label": "07CY9409"},
            {"img_path": "D:/Trial/datasets/val/21BH9525A.jpg", "label": "21BH9525A"},
            {"img_path": "D:/Trial/datasets/val/22BH6517A.jpg", "label": "22BH6517A"},
            {"img_path": "D:/Trial/datasets/val/23BH4962B.jpg", "label": "23BH4962B"},
            {"img_path": "D:/Trial/datasets/val/AN01D4163.jpg", "label": "AN01D4163"},
            {"img_path": "D:/Trial/datasets/val/AN01H0908.jpg", "label": "AN01H0908"},
            {"img_path": "D:/Trial/datasets/val/AN01H1689.jpg", "label": "AN01H1689"},
            {"img_path": "D:/Trial/datasets/val/AN01L6155.jpg", "label": "AN01L6155"},
            {"img_path": "D:/Trial/datasets/val/AN01M4M.jpg", "label": "AN01M4M"},
            {"img_path": "D:/Trial/datasets/val/AN01N7510.jpg", "label": "AN01N7510"},
            {"img_path": "D:/Trial/datasets/val/AN01P9687.jpg", "label": "AN01P9687"},
            {"img_path": "D:/Trial/datasets/val/AP02BP2454.jpg", "label": "AP02BP2454"},
            {"img_path": "D:/Trial/datasets/val/AP03AF2119.jpg", "label": "AP03AF2119"},
            {"img_path": "D:/Trial/datasets/val/AP03BQ2830.jpg", "label": "AP03BQ2830"},
            {"img_path": "D:/Trial/datasets/val/AP03CB4838.jpg", "label": "AP03CB4838"},
            {"img_path": "D:/Trial/datasets/val/AP04AS8753.jpg", "label": "AP04AS8753"},
            {"img_path": "D:/Trial/datasets/val/AP04BA4464.jpg", "label": "AP04BA4464"},
            {"img_path": "D:/Trial/datasets/val/AP04M7627.jpg", "label": "AP04M7627"},
            {"img_path": "D:/Trial/datasets/val/AP05BY7799.jpg", "label": "AP05BY7799"},
            {"img_path": "D:/Trial/datasets/val/AP05DY3395.jpg", "label": "AP05DY3395"},
            {"img_path": "D:/Trial/datasets/val/AP07AA1873.jpg", "label": "AP07AA1873"},
            {"img_path": "D:/Trial/datasets/val/AP07AD5555.jpg", "label": "AP07AD5555"},
            {"img_path": "D:/Trial/datasets/val/AP07AF1581.jpg", "label": "AP07AF1581"},
            {"img_path": "D:/Trial/datasets/val/AP07AN3915.jpg", "label": "AP07AN3915"},
            {"img_path": "D:/Trial/datasets/val/AP07AP1549.jpg", "label": "AP07AP1549"},
            {"img_path": "D:/Trial/datasets/val/AP07AP9347.jpg", "label": "AP07AP9347"},
            {"img_path": "D:/Trial/datasets/val/AP07AQ5346.jpg", "label": "AP07AQ5346"},
            {"img_path": "D:/Trial/datasets/val/AP07AS0917.jpg", "label": "AP07AS0917"},
            {"img_path": "D:/Trial/datasets/val/AP07AT4483.jpg", "label": "AP07AT4483"},
            {"img_path": "D:/Trial/datasets/val/AP07AT9413.jpg", "label": "AP07AT9413"},
            {"img_path": "D:/Trial/datasets/val/AP07AU2455.jpg", "label": "AP07AU2455"},
            {"img_path": "D:/Trial/datasets/val/AP07AU7932.jpg", "label": "AP07AU7932"},
            {"img_path": "D:/Trial/datasets/val/AP07AU8012.jpg", "label": "AP07AU8012"},
            {"img_path": "D:/Trial/datasets/val/AP07AV0964.jpg", "label": "AP07AV0964"},
            {"img_path": "D:/Trial/datasets/val/AP07AV3360.jpg", "label": "AP07AV3360"},
            {"img_path": "D:/Trial/datasets/val/AP07AW0817.jpg", "label": "AP07AW0817"},
            {"img_path": "D:/Trial/datasets/val/AP07AY1069.jpg", "label": "AP07AY1069"},
            {"img_path": "D:/Trial/datasets/val/AP07AY1487.jpg", "label": "AP07AY1487"},
            {"img_path": "D:/Trial/datasets/val/AP07AY6493.jpg", "label": "AP07AY6493"},
            {"img_path": "D:/Trial/datasets/val/AP07AZ3453.jpg", "label": "AP07AZ3453"},
            {"img_path": "D:/Trial/datasets/val/AP07BC2803.jpg", "label": "AP07BC2803"},
            {"img_path": "D:/Trial/datasets/val/AP07BE3183.jpg", "label": "AP07BE3183"},
            {"img_path": "D:/Trial/datasets/val/AP07BE4663.jpg", "label": "AP07BE4663"},
            {"img_path": "D:/Trial/datasets/val/AP07BE6447.jpg", "label": "AP07BE6447"},
            {"img_path": "D:/Trial/datasets/val/AP07BE7428.jpg", "label": "AP07BE7428"},
            {"img_path": "D:/Trial/datasets/val/AP07BH2856.jpg", "label": "AP07BH2856"},
            {"img_path": "D:/Trial/datasets/val/AP07BH4117.jpg", "label": "AP07BH4117"},
            {"img_path": "D:/Trial/datasets/val/AP07BH5208.jpg", "label": "AP07BH5208"},
            {"img_path": "D:/Trial/datasets/val/AP07BJ0544.jpg", "label": "AP07BJ0544"},
            {"img_path": "D:/Trial/datasets/val/AP07BK4503.jpg", "label": "AP07BK4503"},
            {"img_path": "D:/Trial/datasets/val/AP07BM0380.jpg", "label": "AP07BM0380"},
            {"img_path": "D:/Trial/datasets/val/AP07BN6066.jpg", "label": "AP07BN6066"},
            {"img_path": "D:/Trial/datasets/val/AP07BP3220.jpg", "label": "AP07BP3220"},
            {"img_path": "D:/Trial/datasets/val/AP07BP3238.jpg", "label": "AP07BP3238"},
            {"img_path": "D:/Trial/datasets/val/AP07BP6345.jpg", "label": "AP07BP6345"},
            {"img_path": "D:/Trial/datasets/val/AP07BP8315.jpg", "label": "AP07BP8315"},
            {"img_path": "D:/Trial/datasets/val/AP07BP8495.jpg", "label": "AP07BP8495"},
            {"img_path": "D:/Trial/datasets/val/AP2D4014.jpg", "label": "AP2D4014"},
            {"img_path": "D:/Trial/datasets/val/AP37EJ2277.jpg", "label": "AP37EJ2277"},
            {"img_path": "D:/Trial/datasets/val/AP37S7104.jpg", "label": "AP37S7104"},
            {"img_path": "D:/Trial/datasets/val/AP37TA7787.jpg", "label": "AP37TA7787"},
            {"img_path": "D:/Trial/datasets/val/AP39AD6250.jpg", "label": "AP39AD6250"},
            {"img_path": "D:/Trial/datasets/val/AP39AF5028.jpg", "label": "AP39AF5028"},
            {"img_path": "D:/Trial/datasets/val/AP39AO6392.jpg", "label": "AP39AO6392"},
            {"img_path": "D:/Trial/datasets/val/AP39AR8165.jpg", "label": "AP39AR8165"},
            {"img_path": "D:/Trial/datasets/val/AP39BC5530.jpg", "label": "AP39BC5530"},
            {"img_path": "D:/Trial/datasets/val/AP39BC6530.jpg", "label": "AP39BC6530"},
            {"img_path": "D:/Trial/datasets/val/AP39BD0606_.jpg", "label": "AP39BD0606_"},
            {"img_path": "D:/Trial/datasets/val/AP39BM1480.jpg", "label": "AP39BM1480"},
            {"img_path": "D:/Trial/datasets/val/AP39BN1483.jpg", "label": "AP39BN1483"},
            {"img_path": "D:/Trial/datasets/val/AP39BN9989.jpg", "label": "AP39BN9989"},
            {"img_path": "D:/Trial/datasets/val/AP39BR6267.jpg", "label": "AP39BR6267"},
            {"img_path": "D:/Trial/datasets/val/AP39BR8697.jpg", "label": "AP39BR8697"},
            {"img_path": "D:/Trial/datasets/val/AP39BS0740.jpg", "label": "AP39BS0740"},
            {"img_path": "D:/Trial/datasets/val/AP39BS7080.jpg", "label": "AP39BS7080"},
            {"img_path": "D:/Trial/datasets/val/AP39BV4740.jpg", "label": "AP39BV4740"}
        ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write entries to file
    with open(output_path, 'w') as f:
        for entry in data_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created dataset file at {output_path} with {len(data_entries)} entries")

def adapt_paths_for_mac(data_entries):
    """
    Adapt Windows paths to Mac paths.
    
    Args:
        data_entries: List of dictionaries with img_path and label keys
        
    Returns:
        List of dictionaries with adapted paths
    """
    adapted_entries = []
    for entry in data_entries:
        # Replace Windows path with Mac path
        mac_path = entry["img_path"].replace('D:/Trial', '/Users/asutoshsabat/Documents/Trial')
        adapted_entries.append({"img_path": mac_path, "label": entry["label"]})
    
    return adapted_entries

def create_directory_structure(base_path):
    """
    Create the directory structure for the dataset.
    
    Args:
        base_path: Base path for the dataset
    """
    # Create val directory
    val_dir = os.path.join(base_path, 'datasets', 'val')
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Created directory structure at {val_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create dataset file for ANPR')
    parser.add_argument('--output', type=str, default='val_label.txt', help='Path to save the dataset file')
    parser.add_argument('--adapt_paths', action='store_true', help='Adapt Windows paths to Mac paths')
    parser.add_argument('--create_dirs', action='store_true', help='Create directory structure')
    args = parser.parse_args()
    
    # Default entries
    data_entries = [
        {"img_path": "D:/Trial/datasets/val/01CC1A0001.jpg", "label": "01CC1A0001"},
        {"img_path": "D:/Trial/datasets/val/07CY9409.jpg", "label": "07CY9409"},
        # ... (all other entries)
    ]
    
    # Adapt paths if requested
    if args.adapt_paths:
        data_entries = adapt_paths_for_mac(data_entries)
    
    # Create directory structure if requested
    if args.create_dirs:
        create_directory_structure('/Users/asutoshsabat/Documents/Trial')
    
    # Create dataset file
    create_dataset_file(args.output, data_entries)

if __name__ == '__main__':
    main()