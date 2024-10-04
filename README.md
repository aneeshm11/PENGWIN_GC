# Solution to the PENGWIN grand-challenge, MICCAI'24 that uses 2d, 2.5d methods to segment 3d volumes

Trained on 4x A5000 for 8 hours to achieve IOUs of 98-99 , 1-5 ASSD/HD95 on test data.

# METHOD:

1) Directly training on a single view and no further reshaping or resizing.
   The training has to be very accurate and very longer as we are relying on 2d data arrays and volumetric information can be lost with inefficent training.

2) Train 3 separate models, one for each view, perform max pooling or other fusion methods to each of the final voxel of the volume by comparing voxels of 3 volumes produced by 3 models. Lesser training time can achieve decent results. 


Key takeaways- Method1 needs more train time and very fast infernece time, Method2 is the opposite of it. 
             - For both methods , all 2d slices were padded to shape (512x512) during forward pass and unpadded after infernece. 
             - Reshaping can also be done, but it does not give much difference in the results produced, padding showed more accurate shapes during inference. 


# DATA:

please refer the .txt file for crisp understanding of 3d volumes if you are a beginner.

# DEPLOYMENT:

To obtain a dockerised version for ready to use purpose, run the following commands in the same order and place the 
weights.pth file in the resources directory, else comment out the necessary line in the ctinference.py file.

```
./test_run.sh
```

the above line can be skipped if you dont want to test , just run this one below line
```
docker build . \
 --platform=linux/amd64 \
 --tag example-algorithm-preliminary-development-phase-ct
```

```
./save.sh example-algorithm-preliminary-development-phase-ct
```

```
./save.sh
```
![train_vis_epoch27_iter1500](https://github.com/user-attachments/assets/5a202d83-ff61-4d40-b199-65fd205b81a3)
