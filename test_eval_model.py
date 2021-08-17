from generator import Generator
from conditional_generator import CondGenerator_UNET
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


pth_to_cond_generator = "GAN_16.08_MODELS2/gen38.pth"

TEM = TestEvalModel("cuda","CGAN")
TEM.load_generator(pth_to_cond_generator)
TEM.test_generation_time()
TEM.generate_images(16)