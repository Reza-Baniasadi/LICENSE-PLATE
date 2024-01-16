import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
from pathlib import Path
import torch
import pytorch_lightning as pl
from deep_utils import mkdir_incremental, CRNNModelTorch, get_logger, TorchUtils, visualize_data_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CRNNDataset
from settings import Config
from torch.nn import CTCLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

torch.backends.cudnn.benchmark = True
