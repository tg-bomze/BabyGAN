import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
from PIL import ImageFilter
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel, load_images
#from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual losses', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')
    parser.add_argument('--data_dir', default='data', help='Directory for storing optional models')
    parser.add_argument('--mask_dir', default='masks', help='Directory for storing optional masks')
    parser.add_argument('--load_last', default='', help='Start with embeddings from directory')
    parser.add_argument('--dlatent_avg', default='', help='Use dlatent from file specified here for truncation instead of dlatent_avg from Gs')
    parser.add_argument('--model_url', default='karras2019stylegan-ffhq-1024x1024.pkl', help='Fetch a StyleGAN model to train on from this URL')
    parser.add_argument('--architecture', default='vgg16_zhang_perceptual.pkl', help='Сonvolutional neural network model from this URL')
    parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--optimizer', default='ggt', help='Optimization algorithm used for optimizing dlatents')

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=256, help='Size of images for the Resnet model', type=int)
    parser.add_argument('--lr', default=0.25, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--decay_rate', default=0.9, help='Decay rate for learning rate', type=float)
    parser.add_argument('--iterations', default=100, help='Number of optimization steps for each batch', type=int)
    parser.add_argument('--decay_steps', default=4, help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
    parser.add_argument('--early_stopping', default=True, help='Stop early once training stabilizes', type=str2bool, nargs='?', const=True)
    parser.add_argument('--early_stopping_threshold', default=0.5, help='Stop after this threshold has been reached', type=float)
    parser.add_argument('--early_stopping_patience', default=10, help='Number of iterations to wait below threshold', type=int)    
    parser.add_argument('--load_effnet', default='data/finetuned_effnet.h5', help='Model to load for EfficientNet approximation of dlatents')
    parser.add_argument('--load_resnet', default='data/finetuned_resnet.h5', help='Model to load for ResNet approximation of dlatents')
    parser.add_argument('--use_preprocess_input', default=True, help='Call process_input() first before using feed forward net', type=str2bool, nargs='?', const=True)
    parser.add_argument('--use_best_loss', default=True, help='Output the lowest loss value found as the solution', type=str2bool, nargs='?', const=True)
    parser.add_argument('--average_best_loss', default=0.25, help='Do a running weighted average with the previous best dlatents found', type=float)
    parser.add_argument('--sharpen_input', default=True, help='Sharpen the input images', type=str2bool, nargs='?', const=True)

    # Loss function options
    parser.add_argument('--use_vgg_loss', default=0.4, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
    parser.add_argument('--use_pixel_loss', default=1.5, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_mssim_loss', default=200, help='Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_l1_penalty', default=0.5, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_discriminator_loss', default=0.5, help='Use trained discriminator to evaluate realism.', type=float)
    parser.add_argument('--use_adaptive_loss', default=False, help='Use the adaptive robust loss function from Google Research for pixel and VGG feature loss.', type=str2bool, nargs='?', const=True)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=str2bool, nargs='?', const=True)
    parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=str2bool, nargs='?', const=True)
    parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

    # Masking params
    parser.add_argument('--load_mask', default=False, help='Load segmentation masks', type=str2bool, nargs='?', const=True)
    parser.add_argument('--face_mask', default=True, help='Generate a mask for predicting only the face area', type=str2bool, nargs='?', const=True)
    parser.add_argument('--use_grabcut', default=True, help='Use grabcut algorithm on the face mask to better segment the foreground', type=str2bool, nargs='?', const=True)
    parser.add_argument('--scale_mask', default=1.4, help='Look over a wider section of foreground for grabcut', type=float)
    parser.add_argument('--composite_mask', default=True, help='Merge the unmasked area back into the generated image', type=str2bool, nargs='?', const=True)
    parser.add_argument('--composite_blur', default=8, help='Size of blur filter to smoothly composite the images', type=int)

    # Video params
    parser.add_argument('--video_dir', default='videos', help='Directory for storing training videos')
    parser.add_argument('--output_video', default=False, help='Generate videos of the optimization process', type=bool)
    parser.add_argument('--video_codec', default='MJPG', help='FOURCC-supported video codec name')
    parser.add_argument('--video_frame_rate', default=24, help='Video frames per second', type=int)
    parser.add_argument('--video_size', default=512, help='Video size in pixels', type=int)
    parser.add_argument('--video_skip', default=1, help='Only write every n frames (1 = write every frame)', type=int)

    args, other_args = parser.parse_known_args()

    args.decay_steps *= 0.01 * args.iterations # Calculate steps as a percent of total iterations

    if args.output_video:
      import cv2
      synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=args.batch_size)

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(args.model_url, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, clipping_threshold=args.clipping_threshold, tiled_dlatent=args.tile_dlatents, model_res=args.model_res, randomize_noise=args.randomize_noise)
    if (args.dlatent_avg != ''):
        generator.set_dlatent_avg(np.load(args.dlatent_avg))

    perc_model = None
    if (args.use_lpips_loss > 0.00000001):
        with dnnlib.util.open_url(args.architecture, cache_dir=config.cache_dir) as f:
            perc_model =  pickle.load(f)
    perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator, discriminator_network)

    ff_model = None

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        if args.output_video:
          video_out = {}
          for name in names:
            video_out[name] = cv2.VideoWriter(os.path.join(args.video_dir, f'{name}.avi'),cv2.VideoWriter_fourcc(*args.video_codec), args.video_frame_rate, (args.video_size,args.video_size))

        perceptual_model.set_reference_images(images_batch)
        dlatents = None
        if (args.load_last != ''): # load previous dlatents for initialization
            for name in names:
                dl = np.expand_dims(np.load(os.path.join(args.load_last, f'{name}.npy')),axis=0)
                if (dlatents is None):
                    dlatents = dl
                else:
                    dlatents = np.vstack((dlatents,dl))
        else:
            if (ff_model is None):
                if os.path.exists(args.load_resnet):
                    from keras.applications.resnet50 import preprocess_input
                    print("Loading ResNet Model:")
                    ff_model = load_model(args.load_resnet)
            if (ff_model is None):
                if os.path.exists(args.load_effnet):
                    import efficientnet
                    from efficientnet import preprocess_input
                    print("Loading EfficientNet Model:")
                    ff_model = load_model(args.load_effnet)
            if (ff_model is not None): # predict initial dlatents with ResNet model
                if (args.use_preprocess_input):
                    dlatents = ff_model.predict(preprocess_input(load_images(images_batch,image_size=args.resnet_image_size)))
                else:
                    dlatents = ff_model.predict(load_images(images_batch,image_size=args.resnet_image_size))
        if dlatents is not None:
            generator.set_dlatents(dlatents)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, use_optimizer=args.optimizer)
        pbar = tqdm(op, leave=False, total=args.iterations)
        vid_count = 0
        best_loss = None
        best_dlatent = None
        avg_loss_count = 0
        if args.early_stopping:
            avg_loss = prev_loss = None
        for loss_dict in pbar:
            if args.early_stopping: # early stopping feature
                if prev_loss is not None:
                    if avg_loss is not None:
                        avg_loss = 0.5 * avg_loss + (prev_loss - loss_dict["loss"])
                        if avg_loss < args.early_stopping_threshold: # count while under threshold; else reset
                            avg_loss_count += 1
                        else:
                            avg_loss_count = 0
                        if avg_loss_count > args.early_stopping_patience: # stop once threshold is reached
                            print("")
                            break
                    else:
                        avg_loss = prev_loss - loss_dict["loss"]
            pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                if best_dlatent is None or args.average_best_loss <= 0.00000001:
                    best_dlatent = generator.get_dlatents()
                else:
                    best_dlatent = 0.25 * best_dlatent + 0.75 * generator.get_dlatents()
                if args.use_best_loss:
                    generator.set_dlatents(best_dlatent)
                best_loss = loss_dict["loss"]
            if args.output_video and (vid_count % args.video_skip == 0):
              batch_frames = generator.generate_images()
              for i, name in enumerate(names):
                video_frame = PIL.Image.fromarray(batch_frames[i], 'RGB').resize((args.video_size,args.video_size),PIL.Image.LANCZOS)
                video_out[name].write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))
            generator.stochastic_clip_dlatents()
            prev_loss = loss_dict["loss"]
        if not args.use_best_loss:
            best_loss = prev_loss
        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        if args.output_video:
            for name in names:
                video_out[name].release()

        # Generate images from found dlatents and save them
        if args.use_best_loss:
            generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_path, img_name in zip(generated_images, generated_dlatents, images_batch, names):
            mask_img = None
            if args.composite_mask and (args.load_mask or args.face_mask):
                _, im_name = os.path.split(img_path)
                mask_img = os.path.join(args.mask_dir, f'{im_name}')
            if args.composite_mask and mask_img is not None and os.path.isfile(mask_img):
                orig_img = PIL.Image.open(img_path).convert('RGB')
                width, height = orig_img.size
                imask = PIL.Image.open(mask_img).convert('L').resize((width, height))
                imask = imask.filter(ImageFilter.GaussianBlur(args.composite_blur))
                mask = np.array(imask)/255
                mask = np.expand_dims(mask,axis=-1)
                img_array = mask*np.array(img_array) + (1.0-mask)*np.array(orig_img)
                img_array = img_array.astype(np.uint8)
                #img_array = np.where(mask, np.array(img_array), orig_img)
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()


if __name__ == "__main__":
    main()
