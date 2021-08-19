import torch
import os
import torchvision
from fire_mask_dataset import DataLoader, IMAGE_DIR, MASK_DIR

# MODEL NAMES
VANILLA_GAN = "GAN"
CGAN = "CGAN"
BASIC_VAE = "VAE"
CVAE = "CVAE"

# NOISE DIMENSIONS
VANILLA_GAN_Z = (100, 1, 1)
CGAN_Z = (1, 8, 8)
CVAE_Z = [100]

NC = 3 # number of channels on models input associated with image BGR channels


def create_latent_vector(batch_size, latant_dim):  # number tuple
    return torch.randn(batch_size,*latant_dim)


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def save_image_batch(output_folder,name,images):
    images = images.view(-1,3,64,64)
    make_dir(output_folder)
    images_path = os.path.join(output_folder,'{}.jpg'.format(name))
    torchvision.utils.save_image(images,images_path,nrow=4,padding=0,pad_value=0)


def save_image_batch_separate(output_folder, name,images):
    images = images.view(-1, 3, 64, 64)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i in range(images.shape[0]):
        images_path = os.path.join(output_folder, '{}_{}.jpg'.format(name,i))
        image = images[i]
        torchvision.utils.save_image(image, images_path)



def create_example_input(batch_size, model_type,device ):
    if model_type == VANILLA_GAN or model_type == BASIC_VAE:
        example_input  = create_latent_vector(batch_size,(100,1,1)).to(device)
        return example_input

    elif model_type == CGAN :
        z = create_latent_vector(batch_size, CGAN_Z).to(device)
        im, masks = iter(DataLoader(IMAGE_DIR, MASK_DIR, batch_size, shuffle=False).get_data_loader()).next()
        masks = masks.to(device)
        return z, masks

    elif model_type == CVAE:
        z = create_latent_vector(batch_size, CVAE_Z).to(device)
        im, masks = iter(DataLoader(IMAGE_DIR, MASK_DIR, batch_size, shuffle=False).get_data_loader()).next()
        masks = masks.to(device)
        return z, masks