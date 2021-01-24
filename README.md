# Generative Adversarial Networks (GANs)

Implementation of DC-GAN and Wasserstein GAN on Mnist Data

### Dependencies


* python 3.5+
* tensorflow >= 2.0
* sklearn



### Executing program

Define number of epochs, batch size and latent dimension

```
python3 dcgan.py --epochs 10 --batch_size 128 --noise_dim 100
python3 wgan.py --epochs 10 --batch_size 128 --noise_dim 100

```
In order to load a model with pretrained weights

```
python3 dcgan.py --load True
python3 wgan.py --load True

```

## Authors

Allon Hammer