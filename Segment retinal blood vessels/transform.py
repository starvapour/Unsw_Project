from albumentations import Normalize,Compose, Resize
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size):
    return Compose(
        [Resize(img_size, img_size),
         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
         ToTensorV2(p=1.0),]
    )


def get_test_transforms(img_size):
    return Compose(
        [Resize(img_size, img_size),
         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
         ToTensorV2(p=1.0),]
    )