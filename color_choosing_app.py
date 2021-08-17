import numpy as np
import cv2 as cv
import sys
import os
import time

""" 
Claass

"""


class ColorChoosingApp:

    def __init__(self,image_name, color_palette_file_name, image_height=0, image_width=0):
        # argments
        self.image_name = image_name
        self.color_palette_file_name = color_palette_file_name
        self.image_height = image_height # requested max height not actual height of the image
        self.image_width = image_width # requested max width not actual width of the image

        self.output_directory = "out_img"

        # image and vector of colors for further filtering
        self.image = self.open_image()
        self.color_palette = self.read_palette()
        print(self.color_palette)
        #adjusting image
        self.scale_down_to_max_h_w()

        # parameters importan in image and color processing
        self.r = 3 # how many pixel rectangel in distance of picekd area
        self.t = 2 # threshhold for cv in range
        self.n = 3 # grabcut iterations

    def open_image(self):
        try:
            image = cv.imread(cv.samples.findFile(self.image_name))

        except:
            print("Couldn't load file.\n ")
            sys.exit()
        print("Image size: {}".format(image.shape[:]))
        return image


    def read_palette(self):

        if not os.path.isfile(self.color_palette_file_name):

            with open(self.color_palette_file_name,'wb'):
                print("Zadany plik z paletą kolorów nie isnieje. Tworzę nowy")
                return np.array([[]])
        else:
            with open(self.color_palette_file_name, "r+b") as f:
                array = list(f.read())
                color_palette = np.array(array).reshape(-1, 3)
                #color_palette = np.unique(color_palette, axis=0)
                print(color_palette.shape[:])
                return color_palette


    def save_palette_to_file(self):
        with open(self.color_palette_file_name,"wb") as f:
            array = np.array(self.color_palette).flatten()
            array = array.astype('uint8')
            array = list(array)
            array = bytearray(array)

            f.write(array)

    def scale_down_to_max_h_w(self):
        width, height, _ = self.image.shape[:]
        print("szerokosc,wysokosc",width,height)

        new_height = height

        print("zadana wysokosc",self.image_height)
        print("zadana szerokosc",self.image_width)
        if self.image_height != 0:
            print("!!!!!!!!!!",self.image_height,height)
            new_height = min(self.image_height,height)
            width = int((width / height) * new_height)# tu error
            
        if self.image_width != 0:
            print("fua")
            new_width = min(self.image_width,width)
            new_height = int((new_height / width) * new_width)
        else:
            new_width = width

        self.image =  cv.resize(self.image, (new_height, new_width), interpolation=cv.INTER_AREA)
        print("rozmiar wynikowy",self.image.shape[:])


    def save_images_results(self):
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

        cv.imwrite("{}/input.jpg".format(self.output_directory),self.image)
        cv.imwrite("{}/palette_mask.jpg".format(self.output_directory),self.palette_mask)
        cv.imwrite("{}/mask_for_grabcut.jpg".format(self.output_directory), self.mask_for_grabcut)
        cv.imwrite("{}/grabcut.jpg".format(self.output_directory), self.grabcut)

    def mouse_action(self, event, x, y, flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x,y)
            temp_rgb_mat = self.image[y - self.r:y + self.r + 1, x - self.r:x + self.r + 1]  # [x-self.r:x+self.r]
            self.append_unique_colors_to_palette(temp_rgb_mat)

    def append_unique_colors_to_palette(self,temp_rgb_mat):
        temp_rgb_mat = np.reshape(temp_rgb_mat, (-1, 3))
        self.color_palette = np.concatenate((self.color_palette, temp_rgb_mat))
        self.color_palette = np.unique(self.color_palette, axis=0)

    def make_palette_mask(self):
        print("type palety kolorów : ",type(self.color_palette),self.color_palette.shape[:])
        combined_mask = cv.inRange(self.image, np.array([255, 255, 255]), np.array([255, 255, 255]))
        for color in self.color_palette:
            combined_mask += cv.inRange(self.image, np.array(color)-self.t, np.array(color)+ self.t)

        self.mask_for_grabcut = combined_mask

        #combined_mask_rgb = cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)
        # self.palette_mask = self.image&combined_mask_rgb
        self.palette_mask = cv.bitwise_and(self.image,self.image,mask=self.mask_for_grabcut)

    def grabcut_step(self):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        rect = (0,0,1,1)
        mask = self.mask_for_grabcut

        maskG = np.zeros(self.image.shape[:2], dtype = np.uint8)
        maskG[mask==0] = cv.GC_PR_BGD  # GC_PR_FGD cv2.GC_BGD
        maskG[mask>=100] = cv.GC_FGD

        cv.grabCut(self.image, maskG, rect, bgdmodel, fgdmodel,self.n,cv.GC_INIT_WITH_MASK)

        mask2 =  np.where((maskG==1) + (maskG==3), 255, 0).astype('uint8')
        self.grabcut = cv.bitwise_and(self.image, self.image, mask=mask2)

        print("grabcut step")

    def save_close(self):
        self.save_palette_to_file()
        self.save_images_results()

    def run(self):

        margin = 30

        # masks and ooutput images
        self.mask_for_grabcut = np.zeros(self.image.shape[:2],dtype = np.uint8)
        self.palette_mask = np.zeros(self.image.shape, np.uint8)
        self.grabcut = np.zeros(self.image.shape, np.uint8)


        # windows creation and positioning
        cv.namedWindow('input')
        cv.namedWindow('mask_for_grabcut')
        cv.namedWindow('palette_mask')
        cv.namedWindow('grabcut')

        cv.moveWindow('input', 0, 0)
        cv.moveWindow('mask_for_grabcut',self.image.shape[1],0)
        cv.moveWindow('palette_mask',0,  self.image.shape[0]+margin)
        cv.moveWindow('grabcut', self.image.shape[1], self.image.shape[0] + margin)


        # assigining controller
        cv.setMouseCallback('input',self.mouse_action)


        while(1):

            cv.imshow('input',self.image)
            cv.imshow('mask_for_grabcut',self.mask_for_grabcut)
            cv.imshow('palette_mask', self.palette_mask)
            cv.imshow('grabcut',self.grabcut)



            key = cv.waitKey(1)

            if key == 27:
                self.save_close()
                break

            elif key == ord('g'):
                s = time.time()
                self.grabcut_step()
                t = time.time() - s
                print("czas wykonania grabcut o iteracjach = {}: [{}]s".format(self.n,t))

            elif key == ord('p'):
                print("Bierząca paleta kolorów\n",self.color_palette)
            elif key == ord('k'):
                t1 = time.process_time()
                self.make_palette_mask()
                t2 = time.process_time()
                print("czas wykonania maski o color palette o dłoguosci {} : [{}]".format(len(self.color_palette),t2-t1))
            elif key == ord('='):
                self.t += 1
                self.make_palette_mask()
                print(self.t)
            elif key == ord('-'):
                self.t -= 1
                self.make_palette_mask()
                print(self.t)









