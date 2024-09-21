from pathlib import Path
from glob import glob
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


######-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


##### if pooling is needed , only then this is to be used 
# def fuse_3d_predictions(axial_pred, sagittal_pred, coronal_pred):
#     """
#     Fuse predictions from three orthogonal views using a voxel-wise voting mechanism.
    
#     Args:
#     axial_pred (torch.Tensor): Prediction from axial view, shape (D, H, W)
#     sagittal_pred (torch.Tensor): Prediction from sagittal view, shape (H, D, W)
#     coronal_pred (torch.Tensor): Prediction from coronal view, shape (H, W, D)
    
#     Returns:
#     torch.Tensor: Fused prediction of shape (D, H, W)
#     """  
#     sagittal_reshaped = sagittal_pred.permute(1, 0, 2) # Reshape sagittal and coronal predictions to match axial shape (D, H, W)
#     coronal_reshaped = coronal_pred.permute(2, 0, 1)
    
#     D, H, W = axial_pred.shape
#     fused_pred = torch.zeros((D, H, W), dtype=torch.long, device=device)
    
#     for d in range(D):
#         for h in range(H):
#             for w in range(W):
#                 votes = [axial_pred[d, h, w], sagittal_reshaped[d, h, w], coronal_reshaped[d, h, w]]
                
#                 # If all predictions agree
#                 if votes[0] == votes[1] == votes[2]:
#                     fused_pred[d, h, w] = votes[0]
#                 else:
#                     # Count occurrences of each class
#                     vote_counts = torch.bincount(torch.tensor(votes, dtype=torch.long), minlength=30)
                    
#                     # Get the most frequent class (in case of tie, argmax returns the smallest index)
#                     fused_pred[d, h, w] = vote_counts.argmax()
    
#     return fused_pred

######-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=30):
        super(UNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = upconv_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = upconv_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = upconv_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = upconv_block(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))
        
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.output_conv(dec1)
        return output

def pad_torch(x, a, b):
    r, c = x.shape  
    temp = torch.zeros((a, b), device=x.device)
    r_start = (a - r) // 2
    c_start = (b - c) // 2
    temp[r_start:r_start+r, c_start:c_start+c] = x
    return temp

def unpad(x, a, b):
    r, c = np.shape(x)
    start = (r - a) // 2
    end = (c - b) // 2
    return x[start: start+a, end:end+b]

def run():
    input_image = load_image_file(
        location=INPUT_PATH / "images/pelvic-fracture-ct",
    )
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_final = UNet(in_channels=1, out_channels=30).to(device)
    model_final = nn.DataParallel(model_final)
    
    if device.type == "cuda":
        model_final.load_state_dict(torch.load(RESOURCE_PATH / "ctweights.pth"))
    else:
        model_final.load_state_dict(torch.load(RESOURCE_PATH / "ctweights.pth", map_location=device))

    input_array = sitk.GetArrayFromImage(input_image)
    
    final_l = []
    for i in range(input_array.shape[0]):
        a, b = input_array.shape[1], input_array.shape[2]
        final_img = pad_torch(torch.tensor(input_array[i, :, :]), 512, 512).unsqueeze(0).unsqueeze(1)
        
        with torch.no_grad():
            final_pred = model_final(final_img.to(device))
            final_pred = torch.argmax(final_pred, dim=1)
            final_l.append(unpad(final_pred.squeeze(0), a, b))

    final = torch.stack(final_l, dim=0)
    output_array = final.cpu().numpy().astype(np.int8)

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/pelvic-fracture-ct-segmentation",
        array=output_array,
        reference_image=input_image
    )
    
    return 0

def load_image_file(*, location):
    input_files = glob(str(location / "*.mha"))
    if not input_files:
        input_files = glob(str(location / "*.tif"))
    if not input_files:
        raise ValueError("No input file found")
    return sitk.ReadImage(input_files[0])

def write_array_as_image_file(*, location, array, reference_image):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    output_image = sitk.GetImageFromArray(array)
    output_image.SetOrigin(reference_image.GetOrigin())
    output_image.SetSpacing(reference_image.GetSpacing())
    output_image.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(
        output_image,
        str(location / f"output{suffix}"),
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run())