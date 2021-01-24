import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import argparse
import gzip
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape, Dropout, Conv2D, \
    Flatten
from matplotlib import pyplot as plt
import PIL
import time
import sys
from IPython import display
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
import os





class ClipConstraint(Constraint):
    """
    A class with a __call__ methods to help us clip the weights of the kernel in the critic layer
    """

    # set clip value when initialized
    def __init__(self, clip_value):
        """
        :param clip_value: float
        """
        self.clip_value = clip_value

    # clip model weights
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


def wasserstein_loss(y_true, y_pred):
    """

    :param y_true: vector of (batch_size,) positive labels -1, negative labels +1
    :param y_pred: vector of (batch_size,) critic output (-inf, inf).

    After applying the Kantorovich-Rubinstein duality on the original Wasserstein loss we get that:
    The critic tries to minimize f(x) - f(g(z)), encouraging real output f(x) to be smaller.
    The generator tries to minimize -f(g(z)), encouraging fake output f(g(z)) to be larger.

    The -1 label will be multiplied by the average score for real images and encourage a larger predicted average,
    and the +1 label will be multiplied by the average score for real images and have no effect, encouraging a smaller predicted average
    :return: float
    """

    return K.mean(y_true * y_pred)


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
    Creates generator  Z ~ p(z) -- > Dense --> Conv (1X1) --> Conv (2X2) upsample --> Conv (2X2) upsample --> tanh activation
    :return: keras.model.Sequential
    """

    weight_init = keras.initializers.RandomNormal(stddev=0.02)
    model = keras.Sequential()
    model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,), kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=weight_init))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # upsample
    model.add(
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # upsample
    # no batch norm in output layer, and use tanh activation instead of leaky relu
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init,
                              activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    # generator is not optimized directly, therefore we do not compile it

    return model


def make_discriminator_model():
    """
    Creates critic    Conv (2X2) downsample --> Conv (2X2) downsample --> Conv (1X1) --> Linear activation

    :return: keras.model.Sequential
    """

    weight_init = keras.initializers.RandomNormal(stddev=0.02)
    constraint = ClipConstraint(0.01)

    model = tf.keras.Sequential()
    # downsample
    model.add(
        Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=weight_init, kernel_constraint=constraint,
               input_shape=[28, 28, 1]))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    # no batch norm in first layer
    # downsample
    model.add(Conv2D(128, (4, 4), strides=(2, 2), kernel_initializer=weight_init, kernel_constraint=constraint,
                     padding='same'))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # downsample
    model.add(Conv2D(256, (4, 4), strides=(1, 1), kernel_initializer=weight_init, kernel_constraint=constraint,
                     padding='same'))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1, activation='linear'))

    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    model.compile(loss=wasserstein_loss, optimizer=discriminator_optimizer)

    return model




def discriminator_loss(real_output, fake_output):
    """
    Wasserstein loss function. The critic wants to make the distance between f(x) and f(G(z)) as large as possible.
    Since we are minimizing the loss , we can also say we want -f(x) + f(G(z)) to be as small as possible
    Pushing f(x) as low as possible (using -1 as true labels)  and f(G(z)) as high as possible (using +1 as false labels)

    :param real_output: a vector of floats  (Batch_size , ) representing f(x)
    :param fake_output: a vector of floats  (Batch_size , ) representing f(G(z))
    :return: a vector of floats (Batch_size , )

    """
    real_loss = wasserstein_loss(-1 * tf.ones_like(real_output), real_output)  # real images get labels of 1s
    fake_loss = wasserstein_loss(tf.ones_like(fake_output), fake_output)  # fake images get labels of 0s
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    """
    Remember the Generator is only interested in fooling the descriminator. Since the critic wants to give low score
    for real images the Generator would wish the critic would give low for fake images. f(G(z)) --> - inf
    We use the reverse labels (give the critic a fake image and true label -1).
    Essentially pushing G's weights in a way that causes f(G(z))-- > -inf , thus minimizing this loss function

    :param fake_output: a vector of floats  (Batch_size , ) representing f(G(z))
    :return: a vector of floats (Batch_size , )

    """
    return wasserstein_loss(- 1 * tf.ones_like(fake_output), fake_output)  # as explained, labels are 1


@tf.function
def train_step(batch_images, k=5, print_log=False):
    """
    train single iteration (k times for critic and 1 time for generator) on a batch of images

    :param batch_images: matrix (batch_size, 28, 28, 1 )
    :param k: int (number of critic updates per single generator update)
    :param print_log: boolean
    :return: None
    """

    for idx in range(k):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])  # z ~ p(z)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)  # G(z)

            real_output = discriminator(batch_images, training=True)  # D(x) or f(x)
            fake_output = discriminator(generated_images, training=True)  # D(G(z)) or f(G(z))

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        # we get the gradients of the generator and discriminator SEPARATELY and update weights SEPARATELY (in an alternating way)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

        # gradient update step
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        if (idx + 1) == k:  # update generator every k steps only and maybe print logs
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            if print_log:
                # print to log
                tf.print(disc_loss, output_stream='file://./logs/discriminator_wgan.txt')
                tf.print(gen_loss, output_stream='file://./logs/generator_wgan.txt')
                # print to console
                tf.print("discriminator error", disc_loss, output_stream=sys.stdout)
                tf.print("generator error", gen_loss, output_stream=sys.stdout)


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
        plt.savefig('./images/wgan_generated_images_inference_mode/inference_mode_images.png')
    else:
        plt.savefig('./images/wgan_images/image_at_epoch_{:04d}.png'.format(current_epoch_num))

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
    plt.savefig('./images/wgan_generated_images_inference_mode/real_images.png')


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def train(dataset, k=5):
    """
      complete training cycle

      :param dataset: tf.data.Dataset()
      :param k:  int number of critic updates before generator update
      :return:
    """

    for epoch in range(EPOCHS):
        print('********')
        print('Epoch {} Out of {}'.format(epoch, EPOCHS))
        print('********')
        start = time.time()

        for idx, batch_images in enumerate(dataset):
            print('Proceesing Batch {} out of {}'.format(idx + 1, NUM_BATCHES))
            print_log = True if idx == 1 else False
            train_step(batch_images, k, print_log=print_log)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, current_epoch_num=epoch, noise=validation_noise, show=False)

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, current_epoch_num=EPOCHS, noise=validation_noise, show=False)


def args_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--noise_dim", type=int, default=100)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
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

    generator_optimizer = tf.keras.optimizers.RMSprop(0.00005)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(0.00005)

    checkpoint_dir = './checkpoints/wgan_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    if load_pretrained:
        generator.load_weights('./checkpoints/wgan_checkpoints/wgan_generator_final_weights.h5')
        generate_and_save_images(generator, noise = validation_noise, show=True)
        show_real_images()
    else:

        train(train_images)
        generator.save_weights('./checkpoints/wgan_checkpoints/wgan_generator_final_weights.h5')



