import argparse
from color_choosing_app import ColorChoosingApp
import sys

parser = argparse.ArgumentParser(description='Arguments passeed to ColorChoosingApp')

parser.add_argument("image", help="path to image file you want to probe")
parser.add_argument("palette", help="path to binary file to store selected colors")
parser.add_argument("-H","--height", help="maximum height of the probed image.If actual image height is bigger, image will be cropped accordingly")
parser.add_argument("-w","--width", help="maximum height of the probed image.If actual image width is bigger, image will be cropped accordingly")
args = parser.parse_args()

if args.height and not args.height.isnumeric():
    print("\"{}\" is not a number".format(args.height))
    sys.exit()

if args.width and not args.width.isnumeric():
    print("\"{}\" is not a number".format(args.width))
    sys.exit()

if args.width == None:
    args.width = 0

if args.height == None:
    args.height = 0

cca = ColorChoosingApp(args.image,args.palette,int(args.height),int(args.width))
cca.run()
