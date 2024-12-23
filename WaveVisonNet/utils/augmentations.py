import albumentations as A
from albumentations.pytorch import ToTensorV2

def transforms_(cfg):
    train_transform = A.Compose([
            A.Resize(cfg.input_size, cfg.input_size),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
    val_transform = A.Compose([
                A.Resize(cfg.input_size, cfg.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])
    return train_transform, val_transform
