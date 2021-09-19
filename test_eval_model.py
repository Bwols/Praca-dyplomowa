from generator import Generator
from discriminator import  Discriminator

from conditional_generator import CondGenerator_UNET, CondGenerator
from conditional_discriminator import  CondDiscriminator
import torch
from constants import *
from fire_mask_dataset import DataLoader, IMAGE_DIR, MASK_DIR
from time import time
import torchvision.utils as utils
from basic_vae import VAEEncoderDecoder
from cond_vae import CondVAEEncoderDecoder
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import cv2
import random

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


def print_model():
    model = Generator()
    dis = Discriminator()
    print("\n",model)
    print("\n", dis)

    vae= VAEEncoderDecoder()
    print("vae\n\n", )
    print(vae)





def get_real_images(batch_size, model_type, shuffle=False):
    if model_type == CVAE or model_type == CGAN:
        return iter(DataLoader(IMAGE_DIR, MASK_DIR, batch_size, shuffle=shuffle).get_data_loader()).next()
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


def generate_interpolated_images_with_vae(batch_size, path_to_vae, folder, image_name):


    encoder_decoder = VAEEncoderDecoder()
    load_generator(encoder_decoder,path_to_vae)
    images = iter(DataLoader(IMAGE_DIR, None, 2, shuffle=True).get_data_loader()).next()
    images = get_real_images(2, BASIC_VAE)
    z = encoder_decoder.get_latent_vector_from_image(images)
    z = z.view(z.shape[0],1,-1)


    noise_z = create_interpolate_vectors(z[0],z[1],batch_size-2)
    images = encoder_decoder.decode(noise_z)

    make_dir(folder)
    save_image_batch(folder, image_name, images, 16)


def interpolation_experiment(batch_size, path_to_generator, path_to_vae, folder,eks_num):

    generate_interpolated_images_with_generator(batch_size,path_to_generator,folder,"GAN_INTER_{}".format(eks_num))
    generate_interpolated_images_with_vae(batch_size, path_to_vae, folder,"VAE_INTER_{}".format(eks_num))


#generate_interpolated_images_with_vae(16,"vae30.pth","test_vae","interpolation_vae2")


#generate_interpolated_images_with_generator(8,"chosen_generators_vanilla/gen70.pth","test_gan","interpol_dropout")

def generate_images_with_generator(batch_size,path_to_generator,folder,image_name):
    generator = Generator()
    load_generator(generator, path_to_generator)
    noise_z = create_example_input(batch_size, VANILLA_GAN, "cpu")
    images = generator(noise_z)


    generator.eval()
    images2 =generator(noise_z)
    #save images

    make_dir(folder)
    #utils.save_image(images, "{}/{}.png".format(folder, image_name))
    save_image_batch(folder,image_name, images,nrow=4,padding=2)
    save_image_batch(folder, image_name+"eval", images2, nrow=4,padding=2)
    #utils.save_image(images2, "{}/{}.png".format(folder, image_name + "eval"))
    #save_image_batch_separate(folder,image_name,images)



def generate_images_with_cond_generator(batch_size,path_to_generator):
    generator = CondGenerator()
    load_generator(generator, path_to_generator)
    latent, mask=  create_example_input(batch_size, CGAN, "cpu")
    images = generator(latent,mask)

#generate_images_with_cond_generator(16,"chosen_generators_vanilla/gen96_cond.pth")

#generate_images_with_generator(64,"EKSPERYMENTY_GAN/generatorygen80.pth","EKS_GAM","80")
#generate_images_with_generator(64,"chosen_generators_vanilla/gen6.pth","TEST_GAN_IMAGES","6")
#generate_images_with_generator(64,"chosen_generators_vanilla/gen2.pth","TEST_GAN_IMAGES","2")
#generate_images_with_generator(5000,"chosen_generators_vanilla/gen82.pth","IS_GAN5000","IS_TEST")





def calculate_MSE_for_fake_masks(batch_size,n ,path_to_cvae, path_to_cgan ,folder):
    data_loader = DataLoader(IMAGE_DIR, MASK_DIR, batch_size, shuffle=True).get_data_loader()
    cvae = CondVAEEncoderDecoder()
    load_generator(cvae, path_to_cvae)

    cgan = CondGenerator()
    load_generator(cgan, path_to_cgan)

    data_iter = iter(data_loader)
    torch.set_printoptions(profile="full")

    mse = torch.nn.MSELoss(reduction='mean')
    mse_VAE_score = []
    mse_GAN_score = []
    for i in range(n):
        images, real_masks = data_iter.next()


        z = create_latent_vector(batch_size, CVAE_Z)
        fake_cvae_images = cvae.decode((z, real_masks))
        fake_cvae_masks = create_white_mask_tensor(fake_cvae_images)

        mse_error_cvae = mse(fake_cvae_masks.float(), real_masks.float())
        #print("cvae mse:", mse_error_cvae.item())
        mse_VAE_score.append(mse_error_cvae.item())

        z = create_latent_vector(batch_size, CGAN_Z)
        fake_cgan_images = cgan(z, real_masks)
        fake_cgan_masks = create_white_mask_tensor(fake_cgan_images)
        mse_error_cgan = mse(fake_cgan_masks.float(), real_masks.float())
        mse_GAN_score.append(mse_error_cgan.item())



    cvae_mean =  round(np.mean(mse_VAE_score),2)
    cvae_std = round(np.std(mse_VAE_score),2)
    print("Średni błąd odwzorowania maski")
    cvae_mse_line = "CVAE mse: {}  # {}".format(cvae_mean,cvae_std)
    print(cvae_mse_line)

    cgan_mean = round(np.mean(mse_GAN_score), 2)
    cgan_std = round(np.std(mse_GAN_score), 2)


    cgan_mse_line ="CGAN mse: {}  # {}".format(cgan_mean, cgan_std)
    print(cgan_mse_line)


    with open("{}/MSE_n{}.txt".format(folder,batch_size), 'w') as f:
        f.write(cgan_mse_line + "\n")
        f.write(cvae_mse_line)


def calculate_SSIM_for_fake_images(batch_size,path_to_cvae, path_to_cgan ,folder):
    cvae = CondVAEEncoderDecoder()
    load_generator(cvae, path_to_cvae)

    # cgan = CondGenerator_UNET()
    cgan = CondGenerator()
    load_generator(cgan, path_to_cgan)

    real_images, real_masks = get_real_images(batch_size, CVAE, True)

    z = create_latent_vector(batch_size, CVAE_Z)
    fake_cvae_images = cvae.decode((z, real_masks))

    z = create_latent_vector(batch_size, CGAN_Z)
    fake_cgan_images = cgan(z, real_masks)

    real_images_cp = np.transpose(real_images.detach().numpy(), (0, 2, 3, 1))
    fake_cvae_images_cp = np.transpose(fake_cvae_images.detach().numpy(), (0, 2, 3, 1))

    fake_cgan_images_cp = np.transpose(fake_cgan_images.detach().numpy(), (0, 2, 3, 1))

    cgan_score = []
    for i in range(batch_size):
        ssi_score = ssim(real_images_cp[i], fake_cgan_images_cp[i], multichannel=True,
                         data_range=real_images_cp[i].max() - real_images_cp[i].min())
        # print(ssi_score)
        cgan_score.append(ssi_score)
    cgan_ssi_mean_score = np.mean(cgan_score)



    real_images_cp = (real_images_cp + 1) / 2
    cvae_score = []
    for i in range(batch_size):
        ssi_score = ssim(real_images_cp[i], fake_cvae_images_cp[i], multichannel=True,
                         data_range=real_images_cp[i].max() - real_images_cp[i].min())
        # print(ssi_score)
        cvae_score.append(ssi_score)

    cvae_ssi_mean_score = np.mean(cvae_score)
    mean1 = "CVAE mean SSIM score:{} +/- {}".format(round(cvae_ssi_mean_score, 2), round(np.std(cvae_score),2) )+  "   (2 decimals accuracy)"


    mean2 = "CGAN mean SSIM score:{} +/- {}".format(round(cgan_ssi_mean_score, 2), round(np.std(cgan_score),2)) + "   (2 decimals accuracy)"

    print(mean1)
    print(mean2)

    with open("{}/SSIM{}.txt".format(folder,batch_size), 'w') as f:
        f.write(mean1 + "\n")
        f.write(mean2)

def generate_with_generator(path_to_generator,n_samples,output_folder):
    z = create_example_input(n_samples,VANILLA_GAN,"cpu")
    print(z.shape[:])
    generator = Generator()
    load_generator(generator, path_to_generator)
    images = generator(z)
    save_image_batch_separate(output_folder,"GAN",images)

def generate_with_cond_generator(path_to_generator,n_samples,output_folder):
    data = create_example_input(n_samples,CGAN,"cpu",shuffle=True)
    generator = CondGenerator()
    load_generator(generator, path_to_generator)
    images = generator(*data)
    save_image_batch_separate(output_folder, "CGAN", images)


def generate_with_vae(path_to_vae, n_samples, output_folder):
    real_images = get_real_images(n_samples, BASIC_VAE)
    vae = VAEEncoderDecoder()
    load_generator(vae, path_to_vae)
    images, _, _ = vae(real_images)
    save_image_batch_separate(output_folder, "VAE", images)

def generate_with_cvae(path_to_vae, n_samples, output_folder):
    data = create_example_input(n_samples, CVAE, "cpu", shuffle=False)
    cvae = CondVAEEncoderDecoder()
    load_generator(cvae, path_to_vae)
    images = cvae.decode(data)
    save_image_batch_separate(output_folder, "CVAE", images)



def measure_net_time(generator, data,n_samples):
    generator(data) #to warm up GPU :)))
    gen_times = []
    for i in range(n_samples):
        # z = create_example_input(n_samples, VANILLA_GAN, device)
        t1 = time()
        generator(data)

        g_t = time() - t1
       # print(g_t)
        gen_times.append(g_t)

    avg_time = np.mean(gen_times)
    err_time = np.std(gen_times)
    #print("\n", avg_time, err_time)

    return round(avg_time,5), round(err_time,5)

def measure_generator_time(batch_size, path_to_generator, n_samples,  device="cuda"):


    generator = Generator().to(device)
    load_generator(generator, path_to_generator)

    z = create_example_input(batch_size, VANILLA_GAN, device)

    avg_time, err_time = measure_net_time(generator, z, n_samples)

    return  avg_time, err_time


def measure_cond_generator_time(batch_size, path_to_generator, n_samples,  device="cuda"):

    generator = CondGenerator().to(device)
    load_generator(generator, path_to_generator)

    data = create_example_input(batch_size, CGAN, device)
    generator(*data)

    gen_times = []
    for i in range(n_samples):
        # z = create_example_input(n_samples, VANILLA_GAN, device)
        t1 = time()
        generator(*data)

        g_t = time() - t1
       # print(g_t)
        gen_times.append(g_t)

    avg_time = np.mean(gen_times)
    err_time = np.std(gen_times)
    #print("\n",avg_time, err_time)

    return round(avg_time,5), round(err_time,5)




def measure_vae_time(batch_size, path_to_generator, n_samples ,device="cuda"):
    generator = VAEEncoderDecoder().to(device)
    load_generator(generator, path_to_generator)

    data = create_example_input(batch_size, BASIC_VAE, device)

    avg_time, err_time = measure_net_time(generator.decode, data, n_samples)

    return avg_time, err_time

def measure_cvae_time(batch_size, path_to_generator, n_samples ,device="cuda"):
    generator = CondVAEEncoderDecoder().to(device)
    load_generator(generator, path_to_generator)

    data = create_example_input(batch_size, CVAE, device)
    avg_time, err_time = measure_net_time(generator.decode, data, n_samples)
    return avg_time, err_time


def measure_architecture_times(folder):
    gan_t = measure_generator_time(60, "EKSPERYMENT_INTERPOLACJA/gen80.pth", 20)
    cgan_t = measure_cond_generator_time(60, "EKSPERYMENT_WARUNKOWE/gen99.pth", 20 )
    vae_t = measure_vae_time(60, "EKSPERYMENT_INTERPOLACJA/vae99.pth", 20)
    cvae_t = measure_cvae_time(60, "EKSPERYMENT_WARUNKOWE/vae99.pth", 20)

    line_gan = "GAN: {} +/- {}".format(*gan_t)
    #print(line_gan)
    line_cgan = "CGAN: {} +/- {}".format(*cgan_t)
    line_vae = "VAE: {} +/- {}".format(*vae_t)
    line_cvae = "CVAE: {} +/- {}".format(*cvae_t)

    with open("{}/model_times.txt".format(folder), 'w') as f:
        f.write(line_gan + "\n")
        f.write(line_vae + "\n")
        f.write(line_cgan + "\n")
        f.write(line_cvae + "\n")




def generate_whole_set():
    #generate_with_generator("EKSPERYMENT_INTERPOLACJA/gen80.pth",3300,"EKSPERYMENTY_INCEPTION_SCORE/GAN_3300")
    #generate_with_cond_generator("EKSPERYMENT_WARUNKOWE/gen99.pth",3300, "EKSPERYMENTY_INCEPTION_SCORE/CGAN_3300")
    generate_with_vae("EKSPERYMENT_INTERPOLACJA/vae99.pth", 3300,"EKSPERYMENTY_INCEPTION_SCORE/VAE_3300")
    #generate_with_cvae("EKSPERYMENT_WARUNKOWE/vae99.pth", 3300,"EKSPERYMENTY_INCEPTION_SCORE/CVAE_3300")


def compare_generated_images_with_oryginals(batch_size, path_to_cvae, path_to_cgan ,folder):

    cvae = CondVAEEncoderDecoder()
    load_generator(cvae, path_to_cvae)

    #cgan = CondGenerator_UNET()
    cgan = CondGenerator()
    load_generator(cgan, path_to_cgan)


    real_images, real_masks = get_real_images(batch_size,CVAE,True)


    z = create_latent_vector(batch_size, CVAE_Z)
    fake_cvae_images = cvae.decode((z, real_masks))
    fake_cvae_images_ed,_,_ = cvae.forward(real_images, real_masks)


    fake_cvae_masks = create_white_mask_tensor(fake_cvae_images)


    z = create_latent_vector(batch_size,CGAN_Z)
    fake_cgan_images = cgan(z, real_masks )
    fake_cgan_masks = create_white_mask_tensor(fake_cgan_images)
    print(fake_cvae_masks)
    print(real_masks)
    mse = torch.nn.MSELoss(reduction='mean')# or Mse


    #mean mse z 10 prób



    """
    test = fake_cvae_masks - real_masks
    test = test.numpy()
    test =test.reshape(1,-1)
    np.set_printoptions(threshold=np.inf)
    print(test.shape[:])
    a = np.sum(np.absolute(test))
    print(a)
    """



    mse_error_cvae = mse(fake_cvae_masks.float(), real_masks.float())
    print("cvae mse:",mse_error_cvae.item())

    mse_error_cgan = mse(fake_cgan_masks.float(), real_masks.float())
    print("cgan mse:", mse_error_cgan.item())


    real_images_cp = np.transpose(real_images.detach().numpy(), (0,2,3,1))
    fake_cvae_images_cp = np.transpose(fake_cvae_images.detach().numpy(), (0,2,3,1))

    fake_cgan_images_cp = np.transpose(fake_cgan_images.detach().numpy(), (0, 2, 3, 1))


    cgan_score = []
    for i in range(batch_size):
        ssi_score = ssim(real_images_cp[i], fake_cgan_images_cp[i], multichannel=True,
                         data_range=real_images_cp[i].max() - real_images_cp[i].min())
        #print(ssi_score)
        cgan_score.append(ssi_score)
    cgan_ssi_mean_score = np.mean(cgan_score)

    print("CGAN mean SSIM score:", round(cgan_ssi_mean_score, 2), "   (2 decimals accuracy)" ,np.std(cgan_score))
    #np.set_printoptions(threshold=np.inf)

    real_images_cp = (real_images_cp +1)/2



    cvae_score = []
    for i in range(batch_size):

        ssi_score = ssim(real_images_cp[i],fake_cvae_images_cp[i], multichannel=True,data_range=real_images_cp[i].max() - real_images_cp[i].min())
        #print(ssi_score)
        cvae_score.append(ssi_score)

    cvae_ssi_mean_score = np.mean(cvae_score)
    print("CVAE mean SSIM score:", round(cvae_ssi_mean_score, 2), "   (2 decimals accuracy)")

    images = torch.cat((real_images, fake_cgan_images), 0)
    images = torch.cat((images, fake_cvae_images), 0)
    save_image_batch(folder, "images.png", images, nrow=8)

    save_image_batch(folder, "fake_cvae_images.png", fake_cvae_images)
    save_image_batch(folder, "fake_cgan_images1.png", fake_cgan_images)


    masks = torch.cat((real_masks,fake_cgan_masks),0)
    masks = torch.cat((masks, fake_cvae_masks),0)

    save_masks(folder, "masks.png", masks,nrow=8)
    #save_masks(folder, "masks2.png", fake_cgan_masks, nrow=8)





#compare_generated_images_with_oryginals(8,"CVAE_MODELS/vae56.pth","CGAN_MODELS/gen99.pth", "MSE_SSI_test")

def create_random_triangle_mask():
    def get_random_position(x1,x2, y1,y2):
        x = random.randint(x1,x2)
        y = random.randint(y1,y2)
        return [x,y]

    def get_position_array():
        pos_arr = [get_random_position(-20,80,-10,80), get_random_position(0,80,0,100), get_random_position(0,64,0,64)]
        return pos_arr

    image = np.zeros((64, 64))

    if random.randint(0,1) > 0.8:
        pts = np.array(get_position_array())
        cv2.fillPoly(image, pts=[pts], color=1)
        pts = np.array(get_position_array())
        cv2.fillPoly(image, pts=[pts], color=1)
    pts = np.array(get_position_array())
    cv2.fillPoly(image, pts=[pts], color=1)

    #cv2.imshow("filledPolygon", image)
    #cv2.waitKey(0)  # waits until a key is pressed
    #cv2.destroyAllWindows()  # d

    return image



def create_triangle_masks(batch_size):
    masks = np.array([create_random_triangle_mask() for i in range(batch_size)])
    masks = torch.tensor(masks, dtype=torch.float)
    masks = masks.reshape(-1, 1, 64, 64)

    return masks


def test_triangle_masks_on_models(batch_size, path_to_cvae, path_to_generator ,folder):


    masks = create_triangle_masks(batch_size)
    masks = (masks - 0.5) *2

    z1 = create_latent_vector(batch_size, CGAN_Z)
    z2 = create_latent_vector(batch_size, CVAE_Z)

    generator = CondGenerator()
    load_generator(generator, path_to_generator)

    vae = CondVAEEncoderDecoder()
    load_generator(vae, path_to_cvae)




    gen_images = generator(z1,masks)
    vae_images = vae.decode((z2,masks))

    save_image_batch(folder, "gan_triangle2.png", gen_images,8)
    save_image_batch(folder, "vae_triangle2.png", vae_images, 8)
    save_masks(folder, "masks_triangle2.png", masks, 8)


def final_animation_test(batch_size, path_to_generator, folder):
    _, masks = iter(DataLoader( "animation/images", "animation/masks", batch_size).get_data_loader()).__next__()
    print(masks.shape[:])
    generator = CondVAEEncoderDecoder()
    load_generator(generator, path_to_generator)

    z = create_latent_vector(batch_size, CVAE_Z)

    images = generator.decode((z,masks))
    save_image_batch(folder, "animation_cvae", images,batch_size)
    save_masks(folder,"animation_masks",masks,batch_size)




