import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt

# File paths for the multi-temporal images
file_paths = ['image1.tif', 'image2.tif', 'image3.tif']

# List to hold the open raster datasets
datasets = []

# Open each image and add to the list
for file_path in file_paths:
    datasets.append(rasterio.open(file_path))

# Merge the images
merged_image, merged_transform = merge(datasets)

# Close the datasets
for ds in datasets:
    ds.close()

# Optional: Visualize a band from the merged image
show((merged_image, 1), transform=merged_transform)

# Optional: Save the merged image
output_path = 'merged_image.tif'
with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=merged_image.shape[1],
    width=merged_image.shape[2],
    count=merged_image.shape[0],
    dtype=merged_image.dtype,
    crs=datasets[0].crs,
    transform=merged_transform,
) as dst:
    for i in range(merged_image.shape[0]):
        dst.write(merged_image[i, :, :], i + 1)

print("Merged image saved as:", output_path)