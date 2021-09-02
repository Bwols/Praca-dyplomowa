from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
#from test_eval_model import  get_real_images
import cv2 as cv
import glob
import time


def read_images(images_folder_path):
    images = [cv.imread(file) for file in glob.glob("{}/*.png".format(images_folder_path))]
    images += [cv.imread(file) for file in glob.glob("{}/*.jpg".format(images_folder_path))]

    return images

def preprocess_images(images):
    images = np.array(images)
    np.random.shuffle(images)
    scaled_images = []
    for image in images:
        image = cv.resize(image, (299, 299), interpolation=cv.INTER_AREA)
        scaled_images.append(image)
    scaled_images = np.array(scaled_images)
    return scaled_images



def inception_score(P_y_x):
    e = 1E-16
    P_y = np.expand_dims(np.mean(P_y_x,0),0)

    KLD =  P_y_x * (np.log(P_y_x + e) - np.log(P_y + e))
    avg_KLD = np.mean(KLD.sum(1))
    IS = np.exp(avg_KLD)
    return  IS

def inception_score_on_dataset(images, no_batches):

    inceptionv3 = InceptionV3()
    print(images.shape[:])
    batch = np.floor(images.shape[0]/no_batches)
    print("batch_size ={}".format(batch))

    scores = []

    for i in range(no_batches):

        i1 = int(i *batch)
        i2 = int(i*batch +batch)
        P_y_x = inceptionv3.predict(images[i1: i2])

        IS = inception_score(P_y_x)
        scores.append(IS)
        print("{}% ".format(int((i+1) /no_batches *100)))
    print("scores:", scores)
    IS_avg, IS_std = np.mean(scores), np.std(scores)
    return IS_avg, IS_std





images = read_images("IS_GAN5000")# (1.8565556, 0.06115279) gen2
#images = read_images("FD_READY_26.07\\fire_images/")#(3.3051689, 0.24302116)
images = preprocess_images(images)


print("images ready")
print("images size:", images.shape[:])


a = time.time()
score = inception_score_on_dataset(images,10)
print(score)
b = time.time()
print(int(b-a),"s")

