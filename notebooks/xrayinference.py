from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from PIL import Image
import tifffile

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")

CATEGORIES: dict[str, int] = {"SA": 1, "LI": 2, "RI": 3}

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=30):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv3(x)
        
        logits = self.outc(x)
        return logits

def neglog_window(image: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    image = np.array(image)
    shape = image.shape
    if len(shape) == 2:
        image = image[np.newaxis, :, :]

    image += image.min(axis=(1, 2), keepdims=True) + epsilon
    image = -np.log(image)

    image_min = image.min(axis=(1, 2), keepdims=True)
    image_max = image.max(axis=(1, 2), keepdims=True)
    if np.any(image_max == image_min):
        print("mapping constant image to 0. This probably indicates the projector is pointed away from the volume.")
        image[:] = 0
    else:
        image = (image - image_min) / (image_max - image_min)

    if np.any(np.isnan(image)):
        print("got NaN values from negative log transform.")

    if len(shape) == 2:
        return image[0]
    else:
        return image

def _shift(category_id: int, fragment_id: int) -> int:
    return 10 * (category_id - 1) + fragment_id

def masks_to_seg(masks: np.ndarray, category_ids: list[int], fragment_ids: list[int]) -> np.ndarray:
    seg = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint32)
    masks = masks.astype(np.uint32)
    for mask, category_id, fragment_id in zip(masks, category_ids, fragment_ids):
        seg = np.bitwise_or(seg, np.left_shift(mask, _shift(category_id, fragment_id)))
    return seg


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNet(n_channels=1, n_classes=30)
    model = nn.DataParallel(model).to(device)

    weights_path = RESOURCE_PATH / "xrayweights.pth"
    
    model.load_state_dict(torch.load(weights_path, map_location=device))

    input_files = list(INPUT_PATH.glob("images/pelvic-fracture-x-ray/*.tif")) + list(INPUT_PATH.glob("images/pelvic-fracture-x-ray/*.tiff"))
    if not input_files:
        raise FileNotFoundError(f"No TIFF files found in {INPUT_PATH}/images/pelvic-fracture-x-ray/")
    
    original_image = tifffile.imread(input_files[0])

    input_array = neglog_window(original_image)
    input_array = torch.from_numpy(input_array).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_array.to(device))

    binary_pred = (output.detach().cpu().numpy()[0] > 0.5).astype(np.uint8)

    masks = []
    category_ids = []
    fragment_ids = []
    for i in range(30):
        if np.any(binary_pred[i]):
            masks.append(binary_pred[i])
            category_ids.append((i // 10) + 1)
            fragment_ids.append((i % 10) + 1)

    encoded_seg = masks_to_seg(np.array(masks), category_ids, fragment_ids)
    output_array = encoded_seg

    output_path = OUTPUT_PATH / "images/pelvic-fracture-x-ray-segmentation/output.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(output_array, photometric='minisblack', metadata={'spacing': 1, 'unit': 'um'}, resolution=(1, 1))
    
    return 0

if __name__ == "__main__":
    raise SystemExit(run())