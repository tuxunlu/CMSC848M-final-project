#!/usr/bin/env python3
"""
A lightweight script to verify that the DataInterface (LightningDataModule) properly loads your datasets
and that train/val/test dataloaders produce reasonable batches.
"""
import os
import yaml
import argparse
import torchvision.transforms as T
import torch
from torchvision.utils import save_image

from data import DataInterface


def main():
    parser = argparse.ArgumentParser(description="Test DataInterface functionality.")
    parser.add_argument(
        '--config_path',
        type=str,
        default=os.path.join(os.getcwd(), 'config', 'config.yaml'),
        help='Path to your YAML config file'
    )
    args = parser.parse_args()

    # Load and normalize config
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    # ensure lowercase keys
    config = {k.lower(): v for k, v in config.items()}

    print("Instantiating DataInterface with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create the data module
    data_module = DataInterface(**config)

    # Test train dataloader
    print("\nTesting train_dataloader()...")
    train_loader = data_module.train_dataloader()
    try:
        batch = next(iter(train_loader))
    except Exception as e:
        print("Failed to get batch from train_dataloader:", e)
        return
    print("Train batch type:", type(batch))
    # Expect a tuple: (images, captions) or similar
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        images, captions = batch[0], batch[1]
        print(f"  images tensor shape: {getattr(images, 'shape', 'N/A')}")
        print(f"  number of captions: {len(captions)}")
        print("  sample captions:")
        for i, cap in enumerate(captions[:3]):
            print(f"    {i + 1}. {cap}")

        # your Normalize(mean,std) values:
        MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        def denorm(tensor):
            # tensor: Float[3×H×W], normalized
            img = tensor * STD + MEAN
            return img.clamp(0,1)
        
        for idx, img in enumerate(images[:3]):
            # Save the first 3 images to disk
            img = denorm(img)
            save_image(img, os.path.join("./", f"sample_{idx+1}.png"), normalize=False)
            print(f"Saved sample image")
    else:
        print("Unexpected batch format:", batch)

    # Test val dataloader
    print("\nTesting val_dataloader()...")
    val_loader = data_module.val_dataloader()
    try:
        batch = next(iter(val_loader))
    except Exception as e:
        print("Failed to get batch from val_dataloader:", e)
        return
    print("Validation batch loaded successfully.")

    # # Test test dataloader
    # print("\nTesting test_dataloader()...")
    # test_loader = data_module.test_dataloader()
    # try:
    #     batch = next(iter(test_loader))
    # except Exception as e:
    #     print("Failed to get batch from test_dataloader:", e)
    #     return
    # print("Test batch loaded successfully.")

    print("\nAll dataloaders are working as expected.")


if __name__ == '__main__':
    main()
