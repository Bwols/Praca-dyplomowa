import cv2 as cv
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGE_DIR = "FD_READY_26.07/fire_images"
MASK_DIR = "FD_READY_26.07/fire_masks"


def get_full_path(dir, file):
    return "{}/{}".format(dir, file)


class FireMaskDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.add_mask = True
        self.images = []
        self.masks = []

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)) #
             ])


        if mask_dir == None:
            self.add_mask = False
            print("No mask dataset")

        else:
            print("White_mask_dataset")

        for image_name in os.listdir(image_dir):
            image_path = get_full_path(image_dir, image_name)
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = transform(image)
            self.images.append(image)

            if self.add_mask:
                mask_path = get_full_path(mask_dir, image_name)
                mask = cv.imread(mask_path)
                mask = cv.cvtColor(mask,cv.cv2.COLOR_BGR2GRAY)
                mask = transform(mask)
                self.masks.append(mask)


        self.len = len(self.images)
        print("Loaded {} images of fire \n".format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.add_mask:
            item = (self.images[idx], self.masks[idx])
            return item

        else:
            return self.images[idx]


class DataLoader:

    def __init__(self, image_dir, mask_dir, batch_size=16, shuffle=False):

        dataset = FireMaskDataset(image_dir,mask_dir)
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=True
        )

    def get_data_loader(self):
        return self.data_loader



#dl = DataLoader("FD_READY_ALPHA/fire_images", "FD_READY_ALPHA/fire_masks")
#train_loader = dl.get_data_loadet()

