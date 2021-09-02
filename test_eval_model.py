from generator import Generator
from discriminator import  Discriminator

from conditional_generator import CondGenerator_UNET
from conditional_discriminator import  CondDiscriminator
import torch
from constants import *
from fire_mask_dataset import DataLoader, IMAGE_DIR, MASK_DIR
from time import time
import torchvision.utils as utils


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


#test_discriminator(VANILLA_GAN, 64,"GAN_16.08_MODELS_vanilla/gen8.pth","GAN_16.08_MODELS_vanilla/dis8.pth","cuda")

def load_generator(generator, path_to_generator):
    generator.load_state_dict(torch.load(path_to_generator))

def test_generator_generation_time(device,generator_type, path_to_generator, batch_size):
    example_input = create_example_input(batch_size, generator_type, device)

    if generator_type == VANILLA_GAN:
        generator = Generator().to(device)
    elif generator_type == CGAN:
        generator = CondGenerator_UNET().to(device)

    load_generator(generator, path_to_generator)


    t1 = time()
    if generator_type == CGAN:
        output = generator(*example_input)
    else:
        output = generator(example_input)

    t2 = time()
    gen_time = t2 - t1
    print("time needed to generate with {} on {}:  {}".format(generator_type, device,gen_time ))



def test_times_GAN():
    device = "cpu"
    generator_type = VANILLA_GAN
    path_to_generator = "gen82.pth"
    batch_size = 16
    test_generator_generation_time(device, generator_type, path_to_generator, batch_size)

    device = "cuda"
    generator_type = VANILLA_GAN
    path_to_generator = "gen82.pth"
    batch_size = 16
    test_generator_generation_time(device, generator_type, path_to_generator, batch_size)

    device = "cuda"
    generator_type = CGAN
    path_to_generator = "gen13.pth"
    batch_size = 16
    test_generator_generation_time(device, generator_type, path_to_generator, batch_size)

    device = "cpu"
    generator_type = CGAN
    path_to_generator = "gen13.pth"
    batch_size = 16
    test_generator_generation_time(device, generator_type, path_to_generator, batch_size)



#test_times_GAN()

def test_times_profiler(device,batch_size,generator_type):
    example_input = create_example_input(batch_size, generator_type, device)
    from torch.profiler import profile, record_function, ProfilerActivity
#    generator = Generator().to(device)

    with profile(activities=[ProfilerActivity.CPU],profile_memory=True ,record_shapes=True) as prof:
        with record_function("model_inference"):
            generator = CondGenerator_UNET().to(device)
            generator2 = CondGenerator_UNET().to(device)
            #generator(example_input)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

#test_times_profiler("cpu",32,VANILLA_GAN)






from scipy.stats import gaussian_kde

from sklearn.neighbors import KernelDensity

def compute_average_log_likelihood():
    array = np.ones((64,64))
    print(array.shape[:])
    array[2][3] = 7
    array = array
    kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(array)
    #kde = gaussian_kde(array )
    log_dens = kde.score_samples(array)
    print(log_dens)
    exp = np.exp(log_dens)
    print(exp)
#compute_average_log_likelihood()


def create_interpolate_vectors(vector1, vector2, n ):

    #vector2 = np.array(vector2)
    #vector1 = np.array(vector1)
    distance = vector2 - vector1
    step = distance/(n + 1)

    interpolated_vectors = vector1
    for i in range(n):
        new_inter_vec = vector1 + (step * (i+1))
        interpolated_vectors = torch.cat((interpolated_vectors,new_inter_vec),0)

    #interpolated_vectors.append(vector2)
    interpolated_vectors = torch.cat((interpolated_vectors, vector2), 0)
    return interpolated_vectors


def generate_interpolated_images_with_generator(batch_size,path_to_generator,folder,image_name):
    generator = Generator()
    load_generator(generator, path_to_generator)
    z1 = create_latent_vector(1,(100,1,1))
    z2 = create_latent_vector(1,(100,1,1))
    noise_z = create_interpolate_vectors(z1,z2,batch_size-2)
    images = generator(noise_z)

    alpha = 0.3
    z1 = torch.cat(batch_size*[z1])
    z2 = create_latent_vector(batch_size,(100,1,1))* (alpha)
    noise_z =z1.add(z2)

    images2 = generator(noise_z)


    make_dir(folder)
    #utils.save_image(images, "{}/{}.png".format(folder, image_name))
    save_image_batch(folder, image_name, images,16)
    save_image_batch(folder, image_name +"1", images2, 16)


generate_interpolated_images_with_generator(8,"gen82.pth","test_gan","interpol")

def generate_images_with_generator(batch_size,path_to_generator,folder,image_name):
    generator = Generator()
    load_generator(generator, path_to_generator)
    noise_z = create_example_input(batch_size, VANILLA_GAN, "cpu")
    images = generator(noise_z)

    #save images

    make_dir(folder)
    utils.save_image(images, "{}/{}.png".format(folder, image_name))
    save_image_batch_separate(folder,image_name,images)

#generate_images_with_generator(64,"chosen_generators_vanilla/gen82.pth","TEST_GAN_IMAGES","82")
#generate_images_with_generator(64,"chosen_generators_vanilla/gen6.pth","TEST_GAN_IMAGES","6")
#generate_images_with_generator(64,"chosen_generators_vanilla/gen2.pth","TEST_GAN_IMAGES","2")
#generate_images_with_generator(5000,"chosen_generators_vanilla/gen82.pth","IS_GAN5000","IS_TEST")