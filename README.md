# BabyGAN

![logo](https://raw.githubusercontent.com/tg-bomze/BabyGAN/master/media/logo.png)

**Check how it works online:**
- Russian Language [![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)]()
- English Language [![Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)]()

<p>
StyleGAN-based predictor of children's faces from photographs of theoretical parents. The latent representation is extracted from the input images, after which the algorithm mixes them in certain proportions. The neural network model is based on the GAN architecture. Using latency direction, you can change various parameters: age, face position, emotions and gender.
</p>  

**Based on:** [StyleGAN](https://github.com/NVlabs/stylegan)

**Encoder:** [StyleGAN-Encoder](https://github.com/pbaylies/stylegan-encoder)

![example1](https://raw.githubusercontent.com/tg-bomze/BabyGAN/master/media/example1.JPG)
![example2](https://raw.githubusercontent.com/tg-bomze/BabyGAN/master/media/example2.JPG)
![example3](https://raw.githubusercontent.com/tg-bomze/BabyGAN/master/media/example3.JPG)

## Pre-train Models and dictionaries
Follow the [LINK](https://drive.google.com/drive/folders/1xwqqG0HkLe2AiXxjC-XK8OfvMKT1jBlp) and add shortcut to Drive:

![shortcut](media/mount_eng.png)

The folder structure should be:
    
    .
    ├── data                    
    │   └── finetuned_resnet.h5 
    ├── karras2019stylegan-ffhq-1024x1024.pkl
    ├── shape_predictor_68_face_landmarks.dat.bz2
    ├── vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    ├── vgg16_zhang_perceptual.pkl
    └── ...

## Prerequisites
* 64-bit Python 3.6 installation.
* TensorFlow 1.10.0 with GPU support.
* One or more high-end NVIDIA GPUs with at least 11GB of DRAM.
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.

## Generating latent representation of your images
You can generate latent representations of your own images using two scripts:
1) Create folders for photos
> mkdir raw_images aligned_images

2) Extract and align faces from images
> python align_images.py raw_images/ aligned_images/

3) Find latent representation of aligned images
> python encode_images.py aligned_images/ generated_images/ latent_representations/

## Usage BabyGAN
- SOON
