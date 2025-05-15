import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T

# Hugging Face CLIP
from transformers import CLIPTokenizer, CLIPModel

class Coco(Dataset):
    """
    COCO Caption Dataset 2017, returning (image_tensor, clip_text_embedding).

    Args:
        img_dir (str): Path to folder containing train2017/val2017.
        ann_dir (str): Path to folder containing captions_<split>2017.json.
        transform (callable, optional): Image transform. Defaults to Resize+ToTensor+Normalize.
        purpose (str): 'train' or 'validation'. Determines split.
        clip_model_name (str): HuggingFace model ID for CLIP. Defaults to "openai/clip-vit-base-patch32".
        device (str or torch.device): where to run CLIP (e.g. "cpu" or "cuda").
    """
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        transform=None,
        purpose: str = 'train',
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        self.purpose = purpose.lower()
        split = 'train' if self.purpose == 'train' else 'val'

        # Image paths & annotations
        self.img_dir  = os.path.join(img_dir,  f"{split}2017")
        self.ann_file = os.path.join(ann_dir, f"captions_{split}2017.json")
        self.coco     = COCO(self.ann_file)
        self.ann_ids  = sorted(self.coco.anns.keys())

        # Image transforms
        if transform is None:
            self.transform = T.Compose([
                T.Resize((160, 120)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],
                            std =[0.229,0.224,0.225])
            ])
        else:
            self.transform = transform

        # CLIP setup
        self.device    = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip      = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip.eval()  # no dropout / grads

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):
        # 1) get raw data
        ann_id = self.ann_ids[idx]
        ann    = self.coco.anns[ann_id]
        caption= ann['caption']
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # 2) load & transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 3) tokenize caption with CLIP
        token_ids = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).to(self.device)

        return image, token_ids['input_ids'].squeeze(0), token_ids['attention_mask'].squeeze(0)
