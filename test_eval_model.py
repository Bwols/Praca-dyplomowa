from generator import Generator
from discriminator import  Discriminator

from conditional_generator import CondGenerator_UNET
from conditional_discriminator import  CondDiscriminator
import torch
from constants import *
from fire_mask_dataset import DataLoader, IMAGE_DIR, MASK_DIR
from time import time

class TestEvalModel:

    def __init__(self,device, model_type):
        self.device = device
        self.model_type = model_type

        self.generator = None
        self.set_generator()

        self.example_input = self.get_example_input(16)
        print(self.example_input[0].shape[:])
        print(self.example_input[1].shape[:])

    def set_generator(self):
        if self.model_type == VANILLA_GAN:
            self.generator = Generator().to(self.device)
        elif self.model_type == CGAN:
            self.generator = CondGenerator_UNET().to(self.device)

    def load_generator(self, generator_path ):
        self.generator.load_state_dict(torch.load(generator_path))

    def get_example_input(self, batch_size):
        if self.model_type == VANILLA_GAN:
            return create_latent_vector(batch_size,VANILLA_GAN_Z)

        elif self.model_type == CGAN:
            z = create_latent_vector(batch_size, CGAN_Z).to(self.device)
            im, masks = iter(DataLoader(IMAGE_DIR, MASK_DIR, batch_size, shuffle=True).get_data_loader()).next()
            masks = masks.to(self.device)
            return z, masks


    def generate_images(self, images_no):

        images = self.generator(*self.example_input)
        save_image_batch("test_folder","dupa",images)

    def test_generation_time(self):
        t1 = time()
        self.generator(*self.example_input)
        t2 = time()
        t = t2-t1
        print("czas generatora: {}s".format(t))

"""
pth_to_cond_generator = "GAN_16.08_MODELS2/gen38.pth"

TEM = TestEvalModel("cuda","CGAN")
TEM.load_generator(pth_to_cond_generator)
TEM.test_generation_time()
TEM.generate_images(16)
"""




def get_real_images(batch_size, model_type):
    if model_type == CVAE or model_type == CGAN:
        return iter(DataLoader(IMAGE_DIR, MASK_DIR, batch_size, shuffle=False).get_data_loader()).next()
    else:
        return iter(DataLoader(IMAGE_DIR, None, batch_size, shuffle=True).get_data_loader()).next()


def test_discriminator(model_type, batch_size, generator_path, discriminator_path, device):


    if model_type == VANILLA_GAN:
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

    elif model_type == CGAN:
        generator = CondGenerator_UNET().to(device)
        discriminator = CondDiscriminator().to(device)

    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))

    real_images = get_real_images(batch_size, model_type)

    if model_type == CGAN:
        real_images, masks = real_images
        masks = masks.to(device)
    real_images = real_images.to(device)

    gen_input = create_example_input(batch_size, model_type, device)
    fake_images = generator(gen_input)

    real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)



    error_on_real_samples = get_discriminator_error(discriminator, real_images, real_labels)
    print("discriminator L1 error on real images {}".format(error_on_real_samples))

    error_on_fake_samples = get_discriminator_error(discriminator, fake_images, fake_labels)
    print("discriminator L1 error on fake images {}".format(error_on_fake_samples))



def get_discriminator_error(discriminator, image_sample, labels):
    loss_criterion = torch.nn.L1Loss()
    result = discriminator(image_sample)
    #
    loss = loss_criterion(labels, result)

    return loss


test_discriminator(VANILLA_GAN, 64,"GAN_16.08_MODELS_vanilla/gen8.pth","GAN_16.08_MODELS_vanilla/dis8.pth","cuda")