#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf


import argparse

parser = argparse.ArgumentParser(description='Train or execute model to generate texture from picture')
parser.add_argument('--training-input', type=str,
                    default="./data/training/input/wood",
                    help='Path where PNG input images with suffix "-edges.png" and "-image.png" for training are stored')
parser.add_argument('--training-output', type=str,
                    default="./data/training/output/wood",
                    help='Path where generated testing images will be stored during training')
parser.add_argument('--evaluation-input', type=str,
                    default="./data/evaluation/input/wood",
                    help='Path where PNG input images with suffix "-edges.png" and "-image.png" for evaluation are stored')
parser.add_argument('--evaluation-output', type=str,
                    default="./data/evaluation/output/wood",
                    help='Path where generated images will be stored')
parser.add_argument('--checkpoints-path', type=str,
                    default="./checkpoints_gan",
                    help='Path where generated checkpoints will be stored')
parser.add_argument('--epochs', type=int,
                    default=100,
                    help='Number of epochs to run the training')
parser.add_argument('--hide-plots', action="store_true",
                    default=False,
                    help='Do not display plots of every image')
parser.add_argument('--skip-training', action="store_true",
                    default=False,
                    help='Skip training step')
parser.add_argument('--skip-evaluation', action="store_true",
                    default=False,
                    help='Skip evaluation step')
parser.add_argument('-f')

args = parser.parse_args()


# Configuration global variables

TRAINING_INPUT_PATH = args.training_input
TRAINING_OUTPUT_PATH = args.training_output
EVALUATION_INPUT_PATH = args.evaluation_input
EVALUATION_OUTPUT_PATH = args.evaluation_output
CHECKPOINTS_PATH = args.checkpoints_path
DISPLAY_PLOTS = not args.hide_plots
EPOCHS = args.epochs
RUN_TRAINING = not args.skip_training
RUN_EVALUATION = not args.skip_evaluation


import os

# Create directories if needed
if RUN_TRAINING:

    if not os.path.exists(TRAINING_OUTPUT_PATH):
        os.makedirs(TRAINING_OUTPUT_PATH)

    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)
        
if RUN_EVALUATION:
    
    if not os.path.exists(EVALUATION_OUTPUT_PATH):
        os.makedirs(EVALUATION_OUTPUT_PATH)


# Auxiliar Functions

def normalize(image):
    image = (image / 127.5) - 1
    return image

def load_image_set(input_dir, id_str):
    input_img = tf.cast(tf.image.decode_png(tf.io.read_file(input_dir + "/" + id_str + "-edges.png"), channels=3), tf.float32)
    input_img = normalize(input_img)
    
    target_img = tf.cast(tf.image.decode_png(tf.io.read_file(input_dir + "/" + id_str + "-image.png"), channels=3), tf.float32)
    target_img = normalize(target_img)

    return input_img, target_img

def load_image_path(input_dir, filename):
    input_img = tf.cast(tf.image.decode_png(tf.io.read_file(input_dir + "/" + filename), channels=3), tf.float32)
    input_img = normalize(input_img)
    
    return input_img


import numpy as np
import glob
import re

# Main

if RUN_TRAINING:
    image_paths = glob.glob(TRAINING_INPUT_PATH + "/*-edges.png")
    image_filenames = list(map(lambda s: s.split("/")[-1], image_paths))
    data_size = len(image_paths)

    p = re.compile('(.+)-edges\.png')
    ids = sorted(list(map(lambda s: p.search(s).group(1), image_filenames)))

    image_size = load_image_set(TRAINING_INPUT_PATH, ids[0])[0].shape
    
    train_size = round(data_size * 0.80)

    np.random.shuffle(ids)

    train_ids = ids[:train_size]
    test_ids = ids[train_size:]

    train_tensors = list(map(lambda i: load_image_set(TRAINING_INPUT_PATH, i), train_ids))
    test_tensors = list(map(lambda i: load_image_set(TRAINING_INPUT_PATH, i), test_ids))
    
if RUN_EVALUATION:
    image_paths = sorted(glob.glob(EVALUATION_INPUT_PATH + "/*-edges.png"))
    image_filenames = list(map(lambda s: s.split("/")[-1], image_paths))
    
    p = re.compile('(.+)-edges\.png')
    ids = sorted(list(map(lambda s: p.search(s).group(1), image_filenames)))

    image_size = load_image_path(EVALUATION_INPUT_PATH, image_filenames[0]).shape

    eval_tensors = list(map(lambda filename: load_image_path(EVALUATION_INPUT_PATH, filename), image_filenames))


if RUN_TRAINING:
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_tensors,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]))
    )
    train_dataset = train_dataset.batch(1)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_tensors,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]))
    )
    test_dataset = test_dataset.batch(1)

if RUN_EVALUATION:
    eval_dataset = tf.data.Dataset.from_generator(
        lambda: eval_tensors,
        output_types=tf.float32,
        output_shapes=tf.TensorShape([None, None, 3])
    )
    eval_dataset = eval_dataset.batch(1)


from tensorflow.keras import *
from tensorflow.keras.layers import *

def downsampler(filters, apply_batch_normalization=True):
    
    result = Sequential()
    
    initializer = tf.random_normal_initializer(0, 0.02)
    
    result.add(Conv2D(
        filters,
        kernel_size = 4,
        strides = 2,
        padding = "same",
        kernel_initializer = initializer,
        use_bias = not apply_batch_normalization
    ))
    
    if apply_batch_normalization:
        result.add(BatchNormalization())
    
    result.add(LeakyReLU(alpha = 0.2))
    
    return result


def upsampler(filters, apply_dropout=False):
    result = Sequential()
    
    initializer = tf.random_normal_initializer(0, 0.02)
    
    result.add(Conv2DTranspose(
        filters,
        kernel_size = 4,
        strides = 2,
        padding = "same",
        kernel_initializer = initializer,
        use_bias = False
    ))
    
    result.add(BatchNormalization())
    
    if apply_dropout:
        result.add(Dropout(0.5))
    
    result.add(ReLU())
    
    return result


def Generator(input_dim1, input_dim2):
    inputs = tf.keras.layers.Input(shape=[input_dim1, input_dim2, 3])
    
    last_layer = Conv2DTranspose(
        filters = 3,
        kernel_size = 4,
        strides = 2,
        padding = "same",
        kernel_initializer = tf.random_normal_initializer(0, 0.02),
        activation = "tanh"
    )

    # Encoder
    l_e1 = downsampler(64, apply_batch_normalization = False)(inputs)
    l_e2 = downsampler(128)(l_e1)
    l_e3 = downsampler(256)(l_e2)
    l_e4 = downsampler(512)(l_e3)
    l_e5 = downsampler(512)(l_e4)
    l_e6 = downsampler(512)(l_e5)
    l_e7 = downsampler(512)(l_e6)
    l_e8 = downsampler(512)(l_e7)

    # Decoder
    l_d1 = upsampler(512, apply_dropout = True)(l_e8)
    l_d2 = upsampler(512, apply_dropout = True)(concatenate([l_d1, l_e7]))
    l_d3 = upsampler(512, apply_dropout = True)(concatenate([l_d2, l_e6]))
    l_d4 = upsampler(512)(concatenate([l_d3, l_e5]))
    l_d5 = upsampler(256)(concatenate([l_d4, l_e4]))
    l_d6 = upsampler(128)(concatenate([l_d5, l_e3]))
    l_d7 = upsampler(64)(concatenate([l_d6, l_e2]))
    
    last = last_layer(l_d7)
    
    return Model(inputs=[inputs], outputs=last)

generator = Generator(image_size[0], image_size[1])


def Discriminator(input_dim1, input_dim2):
    input_img = Input(shape=[input_dim1, input_dim2, 3])
    generated_img = Input(shape=[input_dim1, input_dim2, 3])
                
    l_d1 = downsampler(64, apply_batch_normalization=False)(concatenate([input_img, generated_img]))
    l_d2 = downsampler(128)(l_d1)
    l_d3 = downsampler(256)(l_d2)
    l_d4 = downsampler(512)(l_d3)
        
    last = Conv2D(
        filters = 1,
        kernel_size = 4,
        strides = 2,
        padding = "same",
        kernel_initializer = tf.random_normal_initializer(0, 0.02),
    )(l_d4)
    
    return Model(inputs=[input_img, generated_img], outputs=last)

discriminator = Discriminator(image_size[0], image_size[1])


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_loss = real_loss + generated_loss
    
    return total_loss


def edge_loss(input_image, generated_image, target_image):
    gray_image = tf.cast(tf.image.rgb_to_grayscale((input_image * 0.5 + 0.5) * 255)[0, ..., -1], tf.uint8)
    edges_mask = tf.greater_equal(tf.constant(0, dtype=tf.uint8), gray_image)
        
    acc_error = 0
    for channel_i in range(3):
        acc_error += tf.reduce_mean(tf.abs(tf.boolean_mask(target_image[0, ..., channel_i], edges_mask) -
                                           tf.boolean_mask(generated_image[0, ..., channel_i], edges_mask)))
    acc_error /= 3.0
    
    return acc_error


LAMBDA = 100
GAMMA = 300

def generator_loss(disc_generated_output, generated_output, input_image, target_image):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    l1_loss = tf.reduce_mean(tf.abs(target_image - generated_output))
    e_loss = edge_loss(input_image, generated_output, target_image)
    
    total_loss = gan_loss + (LAMBDA * l1_loss) + (GAMMA * e_loss)
    
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=CHECKPOINTS_PATH, max_to_keep=3)


def generate_images(model, test_input, test_target, save_filename_path="", display_imgs=True):
    prediction = model([test_input], training = True)

    if not save_filename_path == "":
        output_img_data = tf.cast((prediction[0, ...] * 0.5 + 0.5) * 255, tf.uint8)
        tf.keras.preprocessing.image.save_img(save_filename_path, output_img_data, scale=False)
        
    if display_imgs:
        plt.figure(figsize=(10,10))

        display_list = [test_input[0], test_target[0], prediction[0]]
        title = ["Input image", "Ground truth", "Predicted Image"]
    
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis("off")
    
        plt.show()


@tf.function()
def train_step(input_image, target_image):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        generated_image = generator([input_image], training = True)
    
        generated_image_disc = discriminator([generated_image, input_image], training = True)
    
        target_image_disc = discriminator([target_image, input_image], training = True)
    
        disc_loss = discriminator_loss(target_image_disc, generated_image_disc)
    
        gen_loss = generator_loss(generated_image_disc, generated_image, input_image, target_image)
    
        discriminator_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        
        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))
        generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
        


from IPython.display import clear_output
import os

def train(dataset, epochs):
    
    if os.listdir(CHECKPOINTS_PATH):
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    for epoch in range(epochs):
        img_counter = 0
        for input_image, target_image in train_dataset:
            print("epoch %d - train: %d / %d" % (epoch, img_counter, len(train_ids)))
            img_counter += 1
            train_step(input_image, target_image)
        
        clear_output(wait=True)

        for (input_image, target_image), id_str in zip(test_dataset, test_ids):
            output_path = "%s/%s_%d.jpg" % (TRAINING_OUTPUT_PATH, id_str, epoch)
            generate_images(generator, input_image, target_image, save_filename_path=output_path, display_imgs=DISPLAY_PLOTS)
            
        if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            checkpoint_manager.save()


def evaluate(dataset):
    
    if os.listdir(CHECKPOINTS_PATH):
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    
    for test_input, id_str in zip(dataset, ids):
        prediction = generator([test_input], training = True)
        output_img_path = EVALUATION_OUTPUT_PATH + "/%s-prediction.jpg" % (id_str,)
        output_img_data = tf.cast((prediction[0, ...] * 0.5 + 0.5) * 255, tf.uint8)
        tf.keras.preprocessing.image.save_img(output_img_path, output_img_data, scale=False)


if RUN_TRAINING:
    train(train_dataset, EPOCHS)
    
if RUN_EVALUATION:
    evaluate(eval_dataset)




