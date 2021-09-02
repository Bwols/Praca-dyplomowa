from generative_adversarial_nets import GAN
from VAE import VAE
from constants import create_example_input, save_image_batch , CGAN_Z, create_latent_vector
from fire_mask_dataset import DataLoader
import numpy as np
import torch
from constants import *
import torchvision
from PIL import  Image
import cv2
import random


def create_white_mask_from_tensor(image):
    #white_mask = np.where(((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)), 0, 255)
    np.set_printoptions(threshold=np.inf)
    image = image[0].detach().cpu()
    print(image)
    #print(image.shape[:])
    image = np.array(image)
    image = np.transpose(image,(2,1,0))
    #print(image)
    white_mask = np.where(((image[:, :, 0] < 0) & (image[:, :, 1] < 0) & (image[:, :, 2] < 0)), 0, 255)
    print(white_mask)
    print(white_mask.shape[:])
    #new_image = np.where(((image[:,:, :, 0] == 0) & (image[:,:, :, 1] == 0) & (image[:,:, :, 2] == 0)), [0,0,0], [255,255,255])
    #print("nowy shape", new_image.shape[:])
    return white_mask



def create_white_mask_tensor(image):
    image = image.detach().cpu()
    image = np.array(image)
    image = np.transpose(image,(0,2,3,1))

    white_mask = np.where(((image[:,:, :, 0] <= 0) & (image[:,:, :, 1] <= 0) & (image[:,:, :, 2] <= 0)), 0, 1)

    white_mask = torch.tensor(white_mask)
    white_mask = white_mask.reshape(-1,1,64,64)


    return white_mask





def test_create_white_mask():
    dl = DataLoader("FD_READY_26.07/fire_images", "FD_READY_26.07/fire_masks",4,True)
    train_loader = dl.get_data_loader()
    image, mask = iter(train_loader).__next__()
    print("image dim:",image.shape[:])
    print("mask dim:",mask.shape[:])

    prep_mask = create_white_mask_tensor(image) # <--

    loss_crit = torch.nn.L1Loss()

    loss = loss_crit(prep_mask,mask)
    print(loss)





    prep_mask = prep_mask[0]
    prep_mask = np.array(prep_mask.view(64,64),dtype=np.uint8)


    image = np.array(image[0].view(3,64,64))
    image = np.transpose(image, (1,2,0))

    mask = np.array(mask[0].view(64,64),dtype=np.uint8)
    np.set_printoptions(threshold=np.inf)
    #print(mask)
    #print(prep_mask)





    cv2.imshow('sample image', image)
    cv2.imshow('sample mask', mask)
    cv2.moveWindow('sample mask',0, 64)

    cv2.imshow('prep mask',prep_mask)
    cv2.moveWindow('prep mask', 64, 504)

    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # d



#test_create_white_mask()

def create_random_triangle_mask():
    def get_random_position(x1,x2, y1,y2):
        x = random.randint(x1,x2)
        y = random.randint(y1,y2)
        return [x,y]

    def get_position_array():
        pos_arr = [get_random_position(-20,80,-10,80), get_random_position(0,80,0,100), get_random_position(0,64,0,64)]
        return pos_arr


    pts = np.array(get_position_array())
    image = np.zeros((64, 64))
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



def test_conditional_gan(generator_path, batch_size):

    dl = DataLoader("FD_READY_26.07/fire_images", "FD_READY_26.07/fire_masks", batch_size, True)
    train_loader = dl.get_data_loader()
    images, masks = iter(train_loader).__next__()
    masks = masks.to("cuda")
    triangle_masks =create_triangle_masks(batch_size).to("cuda")

    z = create_latent_vector(batch_size, CGAN_Z).to("cuda")
    model = GAN(device="cuda",conditonal_GAN=True)
    model.load_generator(generator_path)

    new_images = model.generator(z, masks)
    new_triangle_images = model.generator(z, triangle_masks)

    save_masks("test_cond","2.png",masks)
    save_masks("test_cond", "3.png", triangle_masks)


    save_image_batch("test_cond", "1.png", new_images)
    save_image_batch("test_cond", "4.png", new_triangle_images)



def test_vae(vae_path, batch_size):

    def get_latent_vector( mu ,log_var):#reparametrization trick
        std = torch.exp(0.5*log_var)
        eps = torch.rand_like(std)
        return eps *std + mu

    device = "cuda"
    dl = DataLoader("FD_READY_26.07/fire_images", "FD_READY_26.07/fire_masks", batch_size, True)
    train_loader = dl.get_data_loader()
    images, masks = iter(train_loader).__next__()
    images = images.to(device)



    model = VAE(device, conditionalVAE=False)
    model.load_encoder_decoder(vae_path)
    images, mu, log_var = model.encode_decode(images)
    z1 =  create_example_input(batch_size,BASIC_VAE,device)
    z_images = model.generate(z1)
    z2 = get_latent_vector(mu, log_var)
    z2_images = model.generate(z2)

    print(mu.size())
    mu1 = mu[0].repeat(batch_size,1)
    log_var1 = log_var[3].repeat(batch_size,1)
    z3 = get_latent_vector(mu1,log_var1)
    z3_images =model.generate(z3)


    print(mu1.size())

    save_image_batch("test_vae", "1.png", images)
    save_image_batch("test_vae", "4.png", z_images)
    save_image_batch("test_vae", "2.png", z2_images)
    save_image_batch("test_vae", "3.png", z3_images)




def test_cond_vae(cond_vae_path, batch_size):
    device = "cuda"
    dl = DataLoader("FD_READY_26.07/fire_images", "FD_READY_26.07/fire_masks", batch_size, True)
    train_loader = dl.get_data_loader()
    images, masks = iter(train_loader).__next__()
    masks = masks.to(device)
    images = images.to(device)
    z = create_latent_vector(batch_size, CVAE_Z).to(device)
    triangle_masks = create_triangle_masks(batch_size).to(device)

    model = VAE(device,conditionalVAE=True)
    model.load_encoder_decoder(cond_vae_path)
    ed_images, mu, logvar = model.encode_decode((images,masks))
    z_images = model.generate((z,masks))
    z2_images = model.generate((z,triangle_masks))

    images = torch.cat((images, z_images),1)
    save_image_batch("test_cvae", "1.png", images)
    save_image_batch("test_cvae", "2.png", z_images)
    save_image_batch("test_cvae", "3.png", z2_images)
    save_image_batch("test_cvae", "ed.png", ed_images)

    save_masks("test_cvae", "4.png", masks)
    save_masks("test_cvae", "5.png", triangle_masks)


#test_conditional_gan("gen13.pth",32)
#test_vae("vae17.pth",32)

test_cond_vae("vae27.pth", 32)
