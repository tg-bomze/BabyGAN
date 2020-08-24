from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#import tensorflow_probability as tfp
#tf.enable_eager_execution()

import os
import bz2
import PIL.Image
from PIL import ImageFilter
import numpy as np
from keras.models import Model
from keras.utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import traceback
import dnnlib.tflib as tflib

def load_images(images_list, image_size=256, sharpen=False):
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB')
      if image_size is not None:
        img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
        if (sharpen):
            img = img.filter(ImageFilter.DETAIL)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

def tf_custom_adaptive_loss(a,b):
    from adaptive import lossfun
    shape = a.get_shape().as_list()
    dim = np.prod(shape[1:])
    a = tf.reshape(a, [-1, dim])
    b = tf.reshape(b, [-1, dim])
    loss, _, _ = lossfun(b-a, var_suffix='1')
    return tf.math.reduce_mean(loss)

def tf_custom_adaptive_rgb_loss(a,b):
    from adaptive import image_lossfun
    loss, _, _ = image_lossfun(b-a, color_space='RGB', representation='PIXEL')
    return tf.math.reduce_mean(loss)

def tf_custom_l1_loss(img1,img2):
  return tf.math.reduce_mean(tf.math.abs(img2-img1), axis=None)

def tf_custom_logcosh_loss(img1,img2):
  return tf.math.reduce_mean(tf.keras.losses.logcosh(img1,img2))

def create_stub(batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class PerceptualModel:
    def __init__(self, args, batch_size=1, perc_model=None, sess=None):
        self.sess = tf.compat.v1.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.lr = args.lr
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.img_size = args.image_size
        self.layer = args.use_vgg_layer
        self.vgg_loss = args.use_vgg_loss
        self.face_mask = args.face_mask
        self.use_grabcut = args.use_grabcut
        self.scale_mask = args.scale_mask
        self.mask_dir = args.mask_dir
        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None
        self.mssim_loss = args.use_mssim_loss
        if (self.mssim_loss <= self.epsilon):
            self.mssim_loss = None
        self.lpips_loss = args.use_lpips_loss
        if (self.lpips_loss <= self.epsilon):
            self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if (self.l1_penalty <= self.epsilon):
            self.l1_penalty = None
        self.adaptive_loss = args.use_adaptive_loss
        self.sharpen_input = args.sharpen_input
        self.batch_size = batch_size
        if perc_model is not None and self.lpips_loss is not None:
            self.perc_model = perc_model
        else:
            self.perc_model = None
        self.ref_img = None
        self.ref_weight = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None
        self.discriminator_loss = args.use_discriminator_loss
        if (self.discriminator_loss <= self.epsilon):
            self.discriminator_loss = None
        if self.discriminator_loss is not None:
            self.discriminator = None
            self.stub = create_stub(batch_size)

        if self.face_mask:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            landmarks_model_path = unpack_bz2('shape_predictor_68_face_landmarks.dat.bz2')
            self.predictor = dlib.shape_predictor(landmarks_model_path)

    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(self, var_name + "_placeholder", tf.compat.v1.placeholder(var_val.dtype, shape=var_val.get_shape()))
        setattr(self, var_name + "_op", var_val.assign(getattr(self, var_name + "_placeholder")))

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(getattr(self, var_name + "_op"), {getattr(self, var_name + "_placeholder"): var_val})

    def build_perceptual_model(self, generator, discriminator=None):
        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.compat.v1.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.lr, incremented_global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])

        if self.discriminator_loss is not None:
            self.discriminator = discriminator

        generated_image_tensor = generator.generated_image
        generated_image = tf.compat.v1.image.resize_nearest_neighbor(generated_image_tensor,
                                                                  (self.img_size, self.img_size), align_corners=True)

        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_weight = tf.get_variable('ref_weight', shape=generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img")
        self.add_placeholder("ref_weight")

        if (self.vgg_loss is not None):
            vgg16 = VGG16(include_top=False, weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(self.img_size, self.img_size, 3)) # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
            self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
            generated_img_features = self.perceptual_model(preprocess_input(self.ref_weight * generated_image))
            self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
            self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
            self.sess.run([self.features_weight.initializer, self.features_weight.initializer])
            self.add_placeholder("ref_img_features")
            self.add_placeholder("features_weight")

        if self.perc_model is not None and self.lpips_loss is not None:
            img1 = tflib.convert_images_from_uint8(self.ref_weight * self.ref_img, nhwc_to_nchw=True)
            img2 = tflib.convert_images_from_uint8(self.ref_weight * generated_image, nhwc_to_nchw=True)

        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            if self.adaptive_loss:
                self.loss += self.vgg_loss * tf_custom_adaptive_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
            else:
                self.loss += self.vgg_loss * tf_custom_logcosh_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
        # + logcosh loss on image pixels
        if (self.pixel_loss is not None):
            if self.adaptive_loss:
                self.loss += self.pixel_loss * tf_custom_adaptive_rgb_loss(self.ref_weight * self.ref_img, self.ref_weight * generated_image)
            else:
                self.loss += self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight * self.ref_img, self.ref_weight * generated_image)
        # + MS-SIM loss on image pixels
        if (self.mssim_loss is not None):
            self.loss += self.mssim_loss * tf.math.reduce_mean(1-tf.image.ssim_multiscale(self.ref_weight * self.ref_img, self.ref_weight * generated_image, 1))
        # + extra perceptual loss on image pixels
        if self.perc_model is not None and self.lpips_loss is not None:
            self.loss += self.lpips_loss * tf.math.reduce_mean(self.perc_model.get_output_for(img1, img2))
        # + L1 penalty on dlatent weights
        if self.l1_penalty is not None:
            self.loss += self.l1_penalty * 512 * tf.math.reduce_mean(tf.math.abs(generator.dlatent_variable-generator.get_dlatent_avg()))
        # discriminator loss (realism)
        if self.discriminator_loss is not None:
            self.loss += self.discriminator_loss * tf.math.reduce_mean(self.discriminator.get_output_for(tflib.convert_images_from_uint8(generated_image_tensor, nhwc_to_nchw=True), self.stub))
        # - discriminator_network.get_output_for(tflib.convert_images_from_uint8(ref_img, nhwc_to_nchw=True), stub)


    def generate_face_mask(self, im):
        from imutils import face_utils
        import cv2
        rects = self.detector(im, 1)
        # loop over the face detections
        for (j, rect) in enumerate(rects):
            """
            Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            """
            shape = self.predictor(im, rect)
            shape = face_utils.shape_to_np(shape)

            # we extract the face
            vertices = cv2.convexHull(shape)
            mask = np.zeros(im.shape[:2],np.uint8)
            cv2.fillConvexPoly(mask, vertices, 1)
            if self.use_grabcut:
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (0,0,im.shape[1],im.shape[2])
                (x,y),radius = cv2.minEnclosingCircle(vertices)
                center = (int(x),int(y))
                radius = int(radius*self.scale_mask)
                mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
                cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
                cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                mask = np.where((mask==2)|(mask==0),0,1)
            return mask

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size, sharpen=self.sharpen_input)
        image_features = None
        if self.perceptual_model is not None:
            image_features = self.perceptual_model.predict_on_batch(preprocess_input(np.array(loaded_image)))
            weight_mask = np.ones(self.features_weight.shape)

        if self.face_mask:
            image_mask = np.zeros(self.ref_weight.shape)
            for (i, im) in enumerate(loaded_image):
                try:
                    _, img_name = os.path.split(images_list[i])
                    mask_img = os.path.join(self.mask_dir, f'{img_name}')
                    if (os.path.isfile(mask_img)):
                        print("Loading mask " + mask_img)
                        imask = PIL.Image.open(mask_img).convert('L')
                        mask = np.array(imask)/255
                        mask = np.expand_dims(mask,axis=-1)
                    else:
                        mask = self.generate_face_mask(im)
                        imask = (255*mask).astype('uint8')
                        imask = PIL.Image.fromarray(imask, 'L')
                        print("Saving mask " + mask_img)
                        imask.save(mask_img, 'PNG')
                        mask = np.expand_dims(mask,axis=-1)
                    mask = np.ones(im.shape,np.float32) * mask
                except Exception as e:
                    print("Exception in mask handling for " + mask_img)
                    traceback.print_exc()
                    mask = np.ones(im.shape[:2],np.uint8)
                    mask = np.ones(im.shape,np.float32) * np.expand_dims(mask,axis=-1)
                image_mask[i] = mask
            img = None
        else:
            image_mask = np.ones(self.ref_weight.shape)

        if len(images_list) != self.batch_size:
            if image_features is not None:
                features_space = list(self.features_weight.shape[1:])
                existing_features_shape = [len(images_list)] + features_space
                empty_features_shape = [self.batch_size - len(images_list)] + features_space
                existing_examples = np.ones(shape=existing_features_shape)
                empty_examples = np.zeros(shape=empty_features_shape)
                weight_mask = np.vstack([existing_examples, empty_examples])
                image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

            images_space = list(self.ref_weight.shape[1:])
            existing_images_space = [len(images_list)] + images_space
            empty_images_space = [self.batch_size - len(images_list)] + images_space
            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            image_mask = image_mask * np.vstack([existing_images, empty_images])
            loaded_image = np.vstack([loaded_image, np.zeros(empty_images_space)])

        if image_features is not None:
            self.assign_placeholder("features_weight", weight_mask)
            self.assign_placeholder("ref_img_features", image_features)
        self.assign_placeholder("ref_weight", image_mask)
        self.assign_placeholder("ref_img", loaded_image)

    def optimize(self, vars_to_optimize, iterations=200, use_optimizer='adam'):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        if use_optimizer == 'lbfgs':
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list=vars_to_optimize, method='L-BFGS-B', options={'maxiter': iterations})
        else:
            if use_optimizer == 'ggt':
                optimizer = tf.contrib.opt.GGTOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
            self.sess.run(tf.variables_initializer(optimizer.variables()))
            fetch_ops = [min_op, self.loss, self.learning_rate]
        #min_op = optimizer.minimize(self.sess)
        #optim_results = tfp.optimizer.lbfgs_minimize(make_val_and_grad_fn(get_loss), initial_position=vars_to_optimize, num_correction_pairs=10, tolerance=1e-8)
        self.sess.run(self._reset_global_step)
        #self.sess.graph.finalize()  # Graph is read-only after this statement.
        for _ in range(iterations):
            if use_optimizer == 'lbfgs':
                optimizer.minimize(self.sess, fetches=[vars_to_optimize, self.loss])
                yield {"loss":self.loss.eval()}
            else:
                _, loss, lr = self.sess.run(fetch_ops)
                yield {"loss":loss,"lr":lr}
