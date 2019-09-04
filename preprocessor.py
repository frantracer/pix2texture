import os
import shutil
import json

import numpy as np

from PIL import Image

import scipy
from skimage import io
from skimage import feature
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import exposure

import argparse

# Argument parsing

parser = argparse.ArgumentParser(
  description='Preprocess images to detect their edges',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
  '--input-path', type=str, required=True,
  default="./notebooks/data/original",
  help='Path where input images are stored')
parser.add_argument(
  '--output-path', type=str, required=True,
  default="./notebooks/data/input",
  help='Path where generated images will be stored')
parser.add_argument(
  '--subdirs', type=str, nargs='+', required=True,
  default=["wood"],
  help='Subdirs in the input path to be processed')
parser.add_argument(
  '--img-width', type=int,
  default=512,
  help='Output image width')
parser.add_argument(
  '--img-height', type=int,
  default=512,
  help='Output image height')
parser.add_argument(
  '--force-rescaling', action="store_true",
  default=False,
  help='Rescale every image with the exactly input shape')
parser.add_argument(
  '--canny-sigma', type=int,
  default=5,
  help='Canny sigma fro edge detection, the greater the least edge detected')
parser.add_argument(
  '--thickness', type=int,
  default=4,
  help='Thickness of the detected edges')
parser.add_argument(
  '--disable-mean-color-bg', action="store_true",
  default=False,
  help='Disable the calculation of the background color as the mean of the input')
parser.add_argument(
  '--color-bg', type=int, nargs='+',
  default=[255, 255, 255],
  help='RGB Background color in case mean background color is disabled')
parser.add_argument(
  '--color-edges', type=int, nargs='+',
  default=[0, 0, 0],
  help='Color of the detected edges')

args = parser.parse_args()

# Configuration

input_dir = args.input_path
output_dir = args.output_path
materials = args.subdirs
img_width = args.img_width
img_height = args.img_height
force_rescaling = args.force_rescaling
canny_sigma = args.canny_sigma
mean_color_background = not args.disable_mean_color_bg
color_background = args.color_bg
color_edges = args.color_edges
thickness = args.thickness

# Auxiliar functions

def thicker_borders(img, thickness=1):
  new_img = np.copy(img)
  orig_img = img
  for k in range(thickness):
    for i in range(1, orig_img.shape[0]):
      for j in range(1, orig_img.shape[1]):
        if orig_img[i][j] == 255:
          new_img[i][j] = 255
          new_img[i-1][j] = 255
          new_img[i][j-1] = 255
          new_img[i-1][j-1] = 255
    orig_img = new_img
  return new_img

def rescale_image(img, min_width, min_height, exact = False):
  if exact: # Exact shape
    img = (resize(img, (min_height, min_width), anti_aliasing=True) * 255).astype(np.uint8)
  else: # Proportional scaling
    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]

    desired_height = img_height
    desired_width = img_width

    if(desired_height < min_height):
      desired_height = min_height
      desired_width = round(min_height * img_width / img_height)

    if(desired_width < min_width):
      desired_width = min_width
      desired_height = round(min_width * img_height / img_width)
    
    if(img_height != desired_height or img_width != desired_width):
      img = (resize(img, (desired_height, desired_width), anti_aliasing=True) * 255).astype(np.uint8)

  return img

def preprocess_image(img):
  if mean_color_background:
    color_r = np.mean(img[:, :, 0])
    color_g = np.mean(img[:, :, 1])
    color_b = np.mean(img[:, :, 2])
  else:
    color_r = color_background[0]
    color_g = color_background[1]
    color_b = color_background[2]

  img = rgb2gray(img)

  img = exposure.equalize_hist(img)
  img = img.astype(np.float64)

  edges = feature.canny(img, sigma=canny_sigma)
  edges = edges.astype(np.float64) * 255
  edges = edges.astype(np.uint8)
  edges = thicker_borders(edges, thickness=thickness)

  is_background = (edges == 0)
  is_edge = (edges == 255)

  edges_r = np.copy(edges)
  edges_g = np.copy(edges)
  edges_b = np.copy(edges)
  edges_r[is_background] = color_r
  edges_r[is_edge] = color_edges[0]
  edges_g[is_background] = color_g
  edges_g[is_edge] = color_edges[1]
  edges_b[is_background] = color_b
  edges_b[is_edge] = color_edges[2]

  edges = np.stack((edges_r, edges_g, edges_b), axis=2)

  edges = edges.astype(np.uint8)

  return edges

def generate_metadata(img, material_type):
  metadata = {}

  metadata["bright"] = np.mean(img)
  metadata["type"] = material_type

  return metadata

def split_image(img, width, height):
  height_counter = round(img.shape[0] / height)
  width_counter = round(img.shape[1] / width)

  height_segment = int((img.shape[0] - height) / height_counter)
  width_segment = int((img.shape[1] - width) / width_counter)

  img_list = []

  for i in range(height_counter):
    for j in range(width_counter):
      temp_img = img[(i * height_segment):(i * height_segment + height), (j * width_segment):(j * width_segment + width) , :]
      img_list.append(temp_img)

  return img_list

def save_dictionary(filename, dictionary):
  fd = open(filename, "w")
  fd.write(json.dumps(dictionary, indent=2))
  fd.close()

def load_dictionary(filename):
  fd = open(filename, "r")
  dictionary = json.loads(fd.read())
  fd.close()
  return dictionary

# MAIN

if __name__ == "__main__":
  for material in materials:
    output_subdir = output_dir + "/" + material
    material_subdir = input_dir + "/" + material

    if(os.path.isdir(output_subdir)):
      shutil.rmtree(output_subdir)
    os.mkdir(output_subdir)

    img_counter = 0
    for _, type_subdirs, _ in os.walk(material_subdir):
      for material_type in type_subdirs:
        type_subdir = material_subdir + "/" + material_type
        for _, _, files in os.walk(type_subdir):
          for file in files:
            input_img_path = "/".join([type_subdir, file])

            print("Analyzing image: %s ..." % (input_img_path,))

            input_img = rescale_image(io.imread(input_img_path), img_width, img_height, force_rescaling)

            for input_img_split in split_image(input_img, img_width, img_height):
            
              output_img_path = "/".join([output_subdir, "%s_%.4d-image.png" % (material, img_counter,) ])
              edges_img_path = "/".join([output_subdir, "%s_%.4d-edges.png" % (material, img_counter,) ])
              metadata_path = "/".join([output_subdir, "%s_%.4d-meta.txt" % (material, img_counter,) ])

              print("\t- Generating %s ..." % (output_img_path,))
              temp_img_split = preprocess_image(input_img_split)

              io.imsave(output_img_path, input_img_split, check_contrast=False)
              io.imsave(edges_img_path, temp_img_split, check_contrast=False)

              metadata = generate_metadata(input_img_split, material_type)
              save_dictionary(metadata_path, metadata)

              img_counter += 1
