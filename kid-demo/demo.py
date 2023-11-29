import glob
import torch
import numpy as np

from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance

print("Packages successfully imported!")

REAL_IMAGES_FOLDER = "real_images/"
FAKE_IMAGES_FOLDER = "fake_images/"

real_images, fake_images = [], []

for filename in glob.glob(REAL_IMAGES_FOLDER + "*"):
    real_images.append(Image.open(filename))

print(f"Loaded {len(real_images)} real images...")

for filename in glob.glob(FAKE_IMAGES_FOLDER + "*"):
    fake_images.append(Image.open(filename))

print(f"Loaded {len(fake_images)} fake images...")

assert len(real_images) == len(fake_images)

real_images, fake_images = torch.Tensor(np.array(real_images)), torch.Tensor(np.array(fake_images))
real_images, fake_images = torch.transpose(real_images, 2, 3), torch.transpose(fake_images, 2, 3)
real_images, fake_images = torch.transpose(real_images, 1, 2), torch.transpose(fake_images, 1, 2)
real_images, fake_images = real_images / 255.0, fake_images / 255.0
print("Successfully loaded images!")

kid = KernelInceptionDistance(subset_size=10, normalize=True, reset_real_features=False) # Change sample size to reflect number of images
kid.update(real_images, real=True)
kid.update(real_images, real=False)
control_mean, control_stddev = kid.compute()

print(f"Control: Mean ({control_mean}), Standard Deviation ({control_stddev})")

kid.reset()
kid.update(fake_images, real=False)
experimental_mean, experimental_stddev = kid.compute()
print(f"Experimental: Mean ({experimental_mean}), Standard Deviation ({experimental_stddev})")