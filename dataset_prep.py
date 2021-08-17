import numpy as np
import os
import cv2 as cv
from time import process_time
from operator import sub
import  time
N = 3 # grabcut iterations
T = 5  # threshol for
IMAGE_SUBDIR = "fire_images"
MASK_SUBDIR = "fire_masks"
"""
prepares d
"""

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def read_palette(color_palette_file_name):
    try:
        with open(color_palette_file_name,"rb") as f:
            color_palette = list(f.read())
            color_palette = np.array(color_palette).reshape(-1,3)
            return color_palette
    except:
        raise FileNotFoundError


def scale_down_to_max_dim(image,max_h,max_w):
    h, w , _ = image.shape[:]
    new_w = w
    new_h = h
    if h > max_h:
        new_h = max_h
        new_w = int((w/h) * max_h)

    if new_w > max_w:
        new_h = int (new_h/new_w * max_w)
        new_w = max_w
    print("     ",image.shape[:])
    #print("new_h: {}, new w: {}".format(new_h,new_w))
    scaled_down = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    return  scaled_down



def grabcut_segmentation(image,color_palette):
    t1 = process_time()
    combined_mask = cv.inRange(image, np.array([255, 255, 255]), np.array([255, 255, 255]))
    for color in color_palette:
        combined_mask += cv.inRange(image, np.array(color) - T, np.array(color) + T)

    #mask = cv.bitwise_and(image, image, mask=combined_mask)

    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask_fg_bg_assigned = np.zeros(image.shape[:2], dtype = np.uint8)

    mask_fg_bg_assigned[combined_mask==0] = cv.GC_PR_BGD
    mask_fg_bg_assigned[combined_mask>=70] = cv.GC_FGD
    #print("przygotowanie do grabuct{}".format(process_time()-t1))
    cv.grabCut(image,mask_fg_bg_assigned,None,bgdmodel,fgdmodel,N,cv.GC_INIT_WITH_MASK)

    mask2 = np.where((mask_fg_bg_assigned==1) + (mask_fg_bg_assigned==3), 255, 0).astype('uint8')
    grabcut_seg = cv.bitwise_and(image, image, mask=mask2)

    return grabcut_seg

def get_image_active_frame(image):

    h, w, _ = image.shape[:]
    min_pos = (h,w)
    max_pos = (0,0)
    zero_value = [0,0,0]

    for i in range(h):
        for j in range(w):
            pixel = image[i][j].tolist()
            if pixel != zero_value:
                min_pos = [min(min_pos[0], i), min(min_pos[1], j)]
                max_pos = [max(max_pos[0], i), max(max_pos[1], j)]

    new_dimensions = (*tuple(map(sub,max_pos,min_pos)),3)
    active_frame = np.zeros(new_dimensions, dtype=np.uint8)

    active_frame[:,:] = image[min_pos[0]:max_pos[0],min_pos[1]:max_pos[1]]

    return active_frame


def crop_to_rect(image, dim):
    h, w , _ = image.shape[:]

    new_w = dim[1]
    new_h = int((h/w) * new_w)

    if h > w:
        new_h = dim[0]
        new_w = int((w/h) * new_h)

    scaled_image = cv.resize(image,(new_w,new_h), interpolation =cv.INTER_AREA)

    cropped_image = np.zeros((*dim,3),dtype = np.uint8)
    cropped_image[:new_h,:new_w] = scaled_image

    return cropped_image

def creata_white_mask(image):
    #white_mask = np.zeros(image.shape[:2], np.uint8)
    #white_mask = np.where(image <= [0,0,0],0, 255)
    white_mask = np.where(((image[:,:,0]==0) & (image[:,:,1]==0) & (image[:,:,2]==0) ),0 , 255)
    np.set_printoptions(threshold=np.inf)
    
    return white_mask


def save_image(image, output_directory, image_name):
    out_file = "{}/{}.png".format(output_directory,image_name)
    cv.imwrite(out_file,image)



def prepare_dataset(color_palette_file_name,input_directory,output_directory,image_dim):

    i = 0
    height, width = image_dim
    make_dir(output_directory)

    image_subdir = "{}/{}".format(output_directory,IMAGE_SUBDIR)
    masks_subdir = "{}/{}".format(output_directory,MASK_SUBDIR)

    make_dir(image_subdir)
    make_dir(masks_subdir)

    color_palette = read_palette(color_palette_file_name)


    for roots, dirs, files in os.walk(input_directory):
        print("---- {}:[{}]directories, [{}]files ---- ".format(roots, len(dirs), len(files)))

        for file in files:
            if not file.endswith(".gif"):
                i +=1
                print("     [{}]: {}\{}".format(i, roots, file))

                image_path = os.path.join(roots,file)
                image = cv.imread(image_path)
                try:
                    scaled_image =  scale_down_to_max_dim(image,200,200)
                    grabcut_seg = grabcut_segmentation(scaled_image,color_palette)
                    active_frame = get_image_active_frame(grabcut_seg)
                    cropped_image = crop_to_rect(active_frame, image_dim)

                    white_mask = creata_white_mask(cropped_image)

                    save_image(cropped_image, image_subdir, i)
                    save_image(white_mask, masks_subdir,i)

                except:
                    print("     [{}]: operacja nie powiodła się {}".format(i,file))







t1 = time.time()
#prepare_dataset("anycol2.bin","rand test20.07","data_prep_delete3",[128,64])
#prepare_dataset("anycol2.bin","FIRE_DATA_SET","FD_READY_26.07",[64,64])
t2 = time.time()

print("directory time executed: {} minut".format((t2-t1)/60))




