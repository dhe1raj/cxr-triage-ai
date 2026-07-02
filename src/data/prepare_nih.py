import pandas as pd
import os
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def find_image_path(image_index, root_folder):
    """Search for an image inside nested folders like images_001/images/"""
    try:
        for folder in os.listdir(root_folder):
            if folder.startswith("images_"):
                potential_path = os.path.join(root_folder, folder, "images", image_index)
                if os.path.exists(potential_path):
                    return potential_path
    except FileNotFoundError:
        pass
    
    fallback_path = os.path.join(root_folder, image_index)
    if os.path.exists(fallback_path):
        return fallback_path
    
    return str(Path(root_folder) / image_index)

def prepare_nih_with_official_splits(
    csv_path: Path,
    image_root: Path,
    train_val_list: Path,
    test_list: Path,
    output_dir: Path
):
    print("Loading NIH metadata...")
    df = pd.read_csv(csv_path)
    
    # Load image index splits
    print("Loading image splits...")
    with open(train_val_list, 'r') as f:
        train_val_images = set(line.strip() for line in f if line.strip())
    
    with open(test_list, 'r') as f:
        test_images = set(line.strip() for line in f if line.strip())
    
    # Create full image paths
    print("Mapping image paths (this handles split folders)...")
    df["image_path"] = df["Image Index"].apply(
        lambda x: find_image_path(x, str(image_root))
    )
    
    # Binary label: 0 = normal, 1 = abnormal
    df["abnormal"] = (df["Finding Labels"] != "No Finding").astype(int)
    
    # Split by Image Index (correctly matching the official splits)
    train_val_df = df[df["Image Index"].isin(train_val_images)]
    test_df = df[df["Image Index"].isin(test_images)]
    
    print(f"Matched Train/Val images: {len(train_val_df)}")
    print(f"Matched Test images: {len(test_df)}")
    
    # Further split train_val into train and val (80/20)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42, stratify=train_val_df["abnormal"]
    )
    
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "nih_binary_train.csv", index=False)
    val_df.to_csv(output_dir / "nih_binary_val.csv", index=False)
    test_df.to_csv(output_dir / "nih_binary_test.csv", index=False)
    
    sample = df.head(5)[["Image Index", "image_path", "abnormal"]]
    print("\nSample paths (verify these exist):")
    print(sample.to_string(index=False))
    
    print("\n✅ Done! Manifests saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NIH Chest X-ray dataset")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--train_val_list", type=Path, required=True)
    parser.add_argument("--test_list", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="./data/manifests")
    
    args = parser.parse_args()
    
    prepare_nih_with_official_splits(
        args.csv,
        args.image_root,
        args.train_val_list,
        args.test_list,
        args.output
    )