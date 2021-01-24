import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
import gzip
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape ,Dropout, Conv2D, Flatten
from matplotlib import pyplot as plt
import PIL
import time
import sys
from IPython import display
import argparse





def load_mnist(path, is_train=True, return_as_iterator=True):
    """ Load MNIST data from `path`

    :param path: str
    :param is_train: bool
    :param return_as_iterator: bool
    :return: tf.data.DataSet / ( np.array (size, 28,28,1), np.array(size,10))

    """


    if is_train:
        prefix = 'train'
    else:
        prefix = 't10k'


    labels_path = os.path.join(path,'{}-labels-idx1-ubyte.gz'.format(prefix))
    images_path = os.path.join(path,'{}-images-idx3-ubyte.gz'.format(prefix))

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28,1)

    images = (images- 127.5)/127.5 #normalize [-1,1]
    labels = keras.utils.to_categorical(labels, 10)
    if return_as_iterator:
        return tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(BATCH_SIZE)
    else:
        return images, labels


def make_generator_model():
    """
    :return: keras.model.Sequential
    """

    weight_init = keras.initializers.RandomNormal(stddev=0.02)
    model = keras.Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,), kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=weight_init))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    #upsample
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    #upsample
    # no batch norm in output layer, and use tanh activation instead of leaky relu
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    #generator is not optimized directly, therefore we do not compile it

    return model


def make_discriminator_model():
    """
    :return: keras.model.Sequential
    """
    weight_init = keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.Sequential()
    # downsample
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=weight_init, input_shape=[28, 28, 1]))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    # no batch norm in first layer
    # downsample
    model.add(Conv2D(128, (4, 4), strides=(2, 2), kernel_initializer=weight_init, padding='same'))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # downsample
    model.add(Conv2D(256, (4, 4), strides=(1, 1), kernel_initializer=weight_init, padding='same'))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1 ,activation='sigmoid'))

    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1= 0.5)


    model.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

    return model










def discriminator_loss(real_output, fake_output):
    """
    Just a simple log loss function the discriminator wants to minimize

    :param real_output: a vector of floats  (Batch_size , ) representing D(x)
    :param fake_output: a vector of floats  (Batch_size , ) representing D(G(z))
    :return: a vector of floats (Batch_size , )

    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) # real images get labels of 1s
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) # fake images get labels of 0s
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    Remember the Generator is only interested in fooling the descriminator. It wished D(G(z)) --> 1
    Therefore are ground truth labels are 1 and we wish to minimize the log loss towards it.
    Essentially pushing G's weights in a way that causes D(G(z))-- > 1 , thus minimizing this loss function

    :param fake_output: a vector of floats  (Batch_size , ) representing D(G(z))
    :return: a vector of floats (Batch_size , )

    """
    return cross_entropy(tf.ones_like(fake_output), fake_output) # as explained, labels are 1




@tf.function
def train_step(batch_images, is_wgan=False, print_log=False):
    """
    train single iteration (one for discriminator and one for generator) on a batch of images
    :param batch_images: matrix (batch_size, 28, 28, 1 )
        :param is_wgan: boolean

    :param print_log: boolean
    :return: None
    """
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM]) # z ~ p(z)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True) #G(z)

      real_output = discriminator(batch_images, training=True) #D(x) or f(x)
      fake_output = discriminator(generated_images, training=True) #D(G(z)) or f(G(z))


      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    # we get the gradients of the generator and discriminator SEPARATELY and update weights SEPARATELY (in an alternating way)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # gradient update step
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    if print_log:
        # print to log
        tf.print(disc_loss, output_stream='file://./logs/discriminator_dcgan.txt')
        tf.print(gen_loss, output_stream='file://./logs/generator_dcgan.txt')
        # print to console
        tf.print("discriminator error", disc_loss, output_stream=sys.stdout)
        tf.print("generator error" , gen_loss, output_stream=sys.stdout)





def generate_and_save_images(generator,  noise, current_epoch_num= 1, show=False):
    """
    takes our generator and a noise matrix (Batch_size, noise_dim) plots an image an saves it
    :param generator: keras.model.Sequential() our generator model
    :param noise: Matrix (Batch_size, noise_dim)
    :param current_epoch: int
    :param show: boolean
    :return: None
    """

    predictions = generator(noise, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray') #take current image, and grey channel --> (28X28) --> inverse pixels back
        plt.axis('off')

    if show:
        plt.show()
        plt.savefig('./images/dcgan_generated_images_inference_mode/inference_mode_images.png')
    else:
        plt.savefig('./images/dcgan_images/image_at_epoch_{:04d}.png'.format(current_epoch_num))


def show_real_images():
    """
    select one image from each class and plot
    :return:
    """
    X, y = load_mnist(PATH, is_train=False, return_as_iterator=False)
    y = np.argmax(y, axis=1)  # turn one hot to single label
    final_indices = np.array([])
    for cls in range(10):
        indices = np.argwhere(y == cls).reshape(-1)  # get all indices that fit the class
        selected_indices = np.random.choice(indices, 1, replace=False)  # select 1 indice of class
        final_indices = np.concatenate([final_indices, selected_indices])  # append to final indices
    final_indices = final_indices.astype(int)
    X_, y_ = X[final_indices, :, :, :], y[final_indices]  # select x,y with final indices for train

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        plt.title('real image class = {}'.format(i))

    plt.show()
    plt.savefig('./images/dcgan_generated_images_inference_mode/real_images.png')





def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))



def train(dataset):
  for epoch in range(EPOCHS):
    print('********')
    print('Epoch {} Out of {}'.format(epoch, EPOCHS))
    print('********')
    start = time.time()

    for idx, batch_images in enumerate(dataset):
        print('Processing Batch {} out of {}'.format(idx +1 , NUM_BATCHES))
        print_log=True if idx==1 else False
        train_step(batch_images, print_log=print_log)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator, current_epoch_num=epoch, noise=validation_noise, show=False)

    checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,current_epoch_num=EPOCHS,noise = validation_noise, show=False)


def args_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--noise_dim", type = int, default=100)
    parser.add_argument("-e", "--epochs", type = int, default=20)
    parser.add_argument("-b", "--batch_size", type= int, default=128)
    parser.add_argument("-l", "--load",  type=bool, help="pretrained model name")
    return parser.parse_args()

if __name__=="__main__":

    args = args_parsing()

    BATCH_SIZE = args.batch_size
    NOISE_DIM = args.noise_dim
    EPOCHS = args.epochs
    if args.load:
        load_pretrained = True
    else:
        load_pretrained = False

    PATH = './data'
    train_images = load_mnist(PATH, is_train=True)



    num_examples_to_generate = 16
    validation_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])
    NUM_BATCHES = len(list(train_images))

    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    checkpoint_dir = './checkpoints/dcgan_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    if load_pretrained:
        generator.load_weights('./checkpoints/dcgan_checkpoints/dcgan_generator_final_weights.h5')
        generate_and_save_images(generator, noise = validation_noise, show=True)
        show_real_images()
    else:

        train(train_images)
        generator.save_weights('./checkpoints/dcgan_checkpoints/dcgan_generator_final_weights.h5')

