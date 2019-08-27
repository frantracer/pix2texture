import os
import shutil
import json

import numpy as np

from PIL import Image

from skimage import io
from skimage import feature
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import exposure

input_dir = "./data/original"
output_dir = "./data/input"
img_width = 512
img_height = 512
canny_sigma = 5

img_counter = 0

def rescale_image(img, min_width, min_height):
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
  img = rgb2gray(img)

  img = exposure.equalize_hist(img)
  img = img.astype(np.float64)

  edges = feature.canny(img, sigma=canny_sigma)
  edges = edges.astype(np.float64) * 255
  edges = edges.astype(np.uint8)

  return edges

def generate_metadata(img):
  metadata = {}
  metadata["bright"] = np.mean(img)
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

if __name__ == "__main__":
  for _, dirs, _ in os.walk(input_dir):
    for subdir in dirs:
      output_subdir = output_dir + "/" + subdir
      input_subdir = input_dir + "/" + subdir

      if(os.path.isdir(output_subdir)):
        shutil.rmtree(output_subdir)
      os.mkdir(output_subdir)

      for _, _, files in os.walk(input_subdir):
        for file in files:
          input_img_path = "/".join([input_subdir, file])

          print("Analyzing image: %s ..." % (input_img_path,))

          input_img = rescale_image(io.imread(input_img_path), img_width, img_height)

          for input_img_split in split_image(input_img, img_width, img_height):
          
            output_img_path = "/".join([output_subdir, "%.4d-image.png" % (img_counter,) ])
            edges_img_path = "/".join([output_subdir, "%.4d-edges.png" % (img_counter,) ])
            metadata_path = "/".join([output_subdir, "%.4d-meta.txt" % (img_counter,) ])

            print("\t- Generating %s ..." % (output_img_path,))
            temp_img_split = preprocess_image(input_img_split)


            io.imsave(output_img_path, input_img_split, check_contrast=False)
            io.imsave(edges_img_path, temp_img_split, check_contrast=False)

            metadata = generate_metadata(input_img_split)
            save_dictionary(metadata_path, metadata)

            img_counter += 1
