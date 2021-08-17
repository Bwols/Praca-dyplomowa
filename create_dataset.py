import argparse
from dataset_prep import prepare_dataset


parser = argparse.ArgumentParser(description='Arguments passed to function preparing dataset')

parser.add_argument("input directory", help="path to directory containing subfolders with fire images")
parser.add_argument("output directory", help="path ot output directory")
parser.add_argument("color palette", help="binary file with list of colors")
parser.add_argument("image_dim", nargs='+', type=int, help="give 2 dimesnions for final image height and width")

args = parser.parse_args()


print(args.image_dim)

h,w = args.image_dim