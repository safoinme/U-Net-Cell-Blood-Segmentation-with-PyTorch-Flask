
from tqdm import tqdm

import torch.optim as optim
from model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH  = 512
PIN_MEMORY = True
LOAD_MODEL = False
LOSS_FN = "DICE_BCE_LOSS"


TRAIN_IMG_DIR = "/content/drive/MyDrive/UNET/images/"
TRAIN_MASK_DIR = "/content/drive/MyDrive/UNET/masks/"
VAL_IMG_DIR = "/content/drive/MyDrive/UNET/val_set/images/"
VAL_MASK_DIR = "/content/drive/MyDrive/UNET/val_set/masks/"





def train_fn(loader, model, optimizer, loss_fn, scaler):


    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):

        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    if LOSS_FN == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()

    if LOSS_FN == "DICE_LOSS":
        loss_fn = DiceLoss()

    if LOSS_FN == "DICE_BCE_LOSS":
        loss_fn = DiceBCELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load(torch.load("/content/drive/MyDrive/UNET/checkpoint.pth.tar"), model)
        score(val_loader, model)

    scaler = torch.cuda.amp.GradScaler()

    best_score =0.0
    for epoch in range(NUM_EPOCHS):
        print("epoch: ", epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        dice_score = score(val_loader, model, device=DEVICE)

        if dice_score > best_score:
            best_score = dice_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save(checkpoint)
            save_predictions_as_imgs(val_loader, model, folder="/content/drive/MyDrive/UNET/saved_images",
                                 device=DEVICE)

    print(" the best score : ", best_score.item())








