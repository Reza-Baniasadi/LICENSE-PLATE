import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainingSettings:
    def __init__(self):
        self.save_path = "./trained_output"
        self.total_epochs = 120
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_mean = [0.485]
        self.image_std = [0.19]
        self.height = 32
        self.width = 100
        self.channels = 1
        self.num_classes = 50
        self.lstm_hidden_size = 256
        self.lstm_input_dim = 256
        self.learning_rate = 1e-3
        self.lr_decay_factor = 0.1
        self.lr_patience_epochs = 5
        self.min_learning_rate = 1e-6
        self.early_stop_patience = 12
        self.model_name = "latest_plate_model"
        self.batch_size = 128
        self.worker_count = 8

    def apply_args(self, args):
        """آپدیت کردن تنظیمات با Namespace یا dict ورودی"""
        if hasattr(args, "__dict__"):
            self.__dict__.update(vars(args))
        elif isinstance(args, dict):
            self.__dict__.update(args)
        else:
            raise ValueError("args must be Namespace or dict")

    @staticmethod
    def pack_batch(batch):
        """ترکیب تصاویر و لیبل‌ها در یک batch"""
        images, labels = zip(*batch)
        images_tensor = torch.stack(images, dim=0)
        return images_tensor, labels

    @staticmethod
    def build_transforms(height, width):
        """ایجاد pipeline آگوژمنتیشن برای تصاویر"""
        return A.Compose([
            A.Resize(height=height, width=width),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.12, rotate_limit=7, p=0.6),
            A.Normalize(mean=[0.485], std=[0.19]),
            ToTensorV2()
        ])
