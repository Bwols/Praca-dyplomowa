from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from test_eval_model import  get_real_images
import cv2 as cv


def calc_inception_score(images, n_split=10):
    processed =images.astype('float32')
    processed = preprocess_input(processed)

    eps = 1E-16
    model = InceptionV3()
    p_yx = model.predict(processed)

    p_y =np.expand_dims(p_yx.mean(0),0)
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    print(sum_kl_d.shape[:])
    print(avg_kl_d)
    is_score = np.exp(avg_kl_d)
    print(is_score)


images = get_real_images(3000,"GAN")
images = np.array(images)
images = np.transpose(images,(0,2,3,1))

scaled_images = []
for image in images:
    image = cv.resize(image, (299,299), interpolation=cv.INTER_AREA)
    #print(image.shape[:])

    #cv.imshow('sample image', image)
    #cv.waitKey(0)
    scaled_images.append(image)
image = images[0]

scaled_images = np.array(scaled_images)
print("images ready")
print("images size:", scaled_images.shape[:])

images = np.ones((2, 299, 299, 3))
calc_inception_score(scaled_images,10)

