import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pathlib import Path
import numpy as np
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

BASE_DIR = Path("data")
CSV_PATH = BASE_DIR / "metadata_all.csv"
CLASSES = ["drydown", "nutrient_deficiency", "planter_skip", "water"]
BATCH_SIZE = 8
EPOCHS = 50
IMG_SIZE = 512
NUM_WORKERS = 4
CHECKPOINT_DIR = Path("checkpoint_efficient_net")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
num_classes = len(CLASSES) + 1
DEVICE_IS_CUDA = torch.cuda.is_available()

class AgriVisionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, classes: list, augmentation=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.classes = classes
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def _read_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        split = row["split"]
        img_name = row["image"]
        img_stem = Path(img_name).stem  # Remove extensão

        # Imagem RGB (pode ser .jpeg, .jpg, etc)
        img_path = self.base_dir / split / "images" / "rgb" / img_name
        img = self._read_image(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Máscaras: procura por .png com mesmo nome base
        for i, cls in enumerate(self.classes, start=1):
            mask_path = self.base_dir / split / "labels" / cls / "images" / f"{img_stem}.png"
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask[m > 0] = i

        if self.augmentation:
            augmented = self.augmentation(image=img, mask=mask)
            img_t = augmented["image"]
            mask_t = augmented["mask"]
        else:
            img_norm = img.astype("float32") / 255.0
            img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
            mask_t = torch.from_numpy(mask).long()

        if isinstance(mask_t, torch.Tensor):
            if mask_t.ndim == 3 and mask_t.shape[0] == 1:
                mask_t = mask_t.squeeze(0)
            mask_t = mask_t.long()
        else:
            mask_t = torch.from_numpy(np.array(mask_t)).long()

        return img_t.float(), mask_t

class SegModule(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-4, weight_decay: float = 1e-5,
                 encoder_name: str = "efficientnet-b4", encoder_weights: str = "imagenet"):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0)
        self.val_miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        miou = self.train_miou(preds, masks)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mIoU", miou, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        miou = self.val_miou(preds, masks)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mIoU", miou, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class SegDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, base_dir, classes, batch_size=8, num_workers=4,
                 train_tfms=None, val_tfms=None):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.base_dir = base_dir
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms

    def setup(self, stage=None):
        self.train_ds = AgriVisionDataset(self.train_df, self.base_dir, self.classes, self.train_tfms)
        self.val_ds = AgriVisionDataset(self.val_df, self.base_dir, self.classes, self.val_tfms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                         num_workers=self.num_workers, pin_memory=DEVICE_IS_CUDA)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                         num_workers=self.num_workers, pin_memory=DEVICE_IS_CUDA)

def get_training_augs(img_size=IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_validation_augs(img_size=IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    set_seeds(42)
    
    # Carrega dados
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Transformações
    train_tfms = get_training_augs()
    val_tfms = get_validation_augs()
    
    # Data module
    dm = SegDataModule(train_df, val_df, BASE_DIR, CLASSES,
                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                      train_tfms=train_tfms, val_tfms=val_tfms)
    
    # Modelo
    model = SegModule(num_classes=num_classes)
    
    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=8)
    lr_cb = LearningRateMonitor()
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        devices=1,
        accelerator="gpu" if DEVICE_IS_CUDA else "cpu",
        precision="16-mixed" if DEVICE_IS_CUDA else "32-true",
        callbacks=[ckpt_cb, early_cb, lr_cb]
    )
    
    # Treino
    trainer.fit(model, dm)
    print("✅ Treino concluído!")