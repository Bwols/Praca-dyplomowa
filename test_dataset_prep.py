import dataset_prep
import cv2 as cv
import os
from timeit import timeit
from time import process_time

def get_time_output_prepare_dataset(image_name, color_palette_file_name, output_directory):

    color_palette = dataset_prep.read_palette(color_palette_file_name)

    image = cv.imread(image_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    dataset_prep.save_image(image,output_directory,"example")
    t1 = process_time()
    scaled_image = dataset_prep.scale_down_to_max_dim(image,600,600)
    t2 = process_time()

    print_method_time("Czas skalowania w dół:{}", t1, t2)


    #image = cv.resize(image, (800, 500), interpolation=cv.INTER_AREA)
    t1 = process_time()
    grabcut_seg = dataset_prep.grabcut_segmentation(scaled_image, color_palette)
    t2 = process_time()

    print_method_time("Czas wykonania segmentacji Grabcut:{}",t1,t2)
    dataset_prep.save_image(grabcut_seg,output_directory,"grabcut_segm")

    t1 = process_time()
    active_frame = dataset_prep.get_image_active_frame(grabcut_seg)
    t2 = process_time()

    print_method_time("Czas uzyskania aktywnej ramki:{}", t1, t2)
    dataset_prep.save_image(active_frame, output_directory, "active_frame")

    t1 = process_time()
    cropped_image = dataset_prep.crop_to_rect(active_frame, (64, 64))
    t2 = process_time()

    print_method_time("Czas przycięcia i przeskalowania w kwadrat:{}", t1, t2)
    dataset_prep.save_image(cropped_image,output_directory, "croped_image")

    t1 = process_time()
    white_mask = dataset_prep.creata_white_mask(cropped_image)
    t2 = process_time()
    print_method_time("Czas białej maski:{}", t1, t2)
    dataset_prep.save_image(white_mask,output_directory,"white_mask")




def print_method_time(text,time1,time2):
    print(text.format(time2-time1))






get_time_output_prepare_dataset("fire2.jpg", "anycol2.bin", "test dataset")
