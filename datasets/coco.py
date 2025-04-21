import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T



class Coco(Dataset):
    """
    COCO Caption Dataset 2017.

    This dataset can switch between train/val/test splits based on the `purpose` parameter
    and will convert images to tensors by default so DataLoader can batch them.

    Args:
        img_dir (str): Path to the folder containing train2017/val2017 images.
        ann_file (str): Path to the JSON annotation file for the train split (e.g. captions_train2017.json).
        transform (callable, optional): Optional transform for images. Defaults to ToTensor().
        target_transform (callable, optional): Optional transform for captions.
        purpose (str, optional): One of {'train', 'validation', 'test'}. Defaults to 'train'.
        **kwargs: Ignored.
    """

    def __init__(
        self,
        img_dir,
        ann_dir,
        transform=None,
        target_transform=None,
        purpose='train',
        **kwargs
    ):
        super().__init__()
        self.purpose = purpose.lower()

        # Derive correct image directory and annotation file for the split
        split_suffix = 'train' if self.purpose == 'train' else 'val'

        self.img_dir = os.path.join(img_dir, f"{split_suffix}2017")
        self.ann_file = os.path.join(ann_dir, f"captions_{split_suffix}2017.json")

        # Set up transforms: default to ToTensor for images
        if transform is None:
            self.transform = T.Compose([
                T.Resize((640, 480)),  # Resize to 640x480
                T.ToTensor(),         # Convert to tensor
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform

        # Load COCO annotations
        self.coco = COCO(self.ann_file)
        self.ann_ids = sorted(self.coco.anns.keys())

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):
        ann_id = self.ann_ids[idx]
        ann = self.coco.anns[ann_id]
        caption = ann['caption']
        img_info = self.coco.loadImgs(ann['image_id'])[0]

        image = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')
        image = self.transform(image)
        if self.target_transform:
            caption = self.target_transform(caption)

        return image, caption
