from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
from scipy import misc
import json
from pspnet import pspnet101_VOC2012

VOC2012_param = {'crop_size': [473, 473],
                 'num_classes': 21,
                 'model': pspnet101_VOC2012}

SAVE_DIR = './output/'
SNAPSHOT_DIR = './'

color_info_file = '/home/melody/develop/caffe-segmentation/misc/palette/pascal_voc.json'
with open(color_info_file) as fd:
    data = json.load(fd)
    palette = np.array(data['palette'][:-1], dtype=np.uint8)
    IMG_MEAN = np.array(data['mean'], dtype=np.float32)
    print(palette)
    print(IMG_MEAN)
    # IMG_MEAN = map(lambda x: np.float32(x), IMG_MEAN)


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='voc2012',
                        choices=['voc2012', 'ade20k', 'cityscapes'])

    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    ext = filename.split('.')[-1]

    if ext.lower() == 'png':
        img = tf.image.decode_png(tf.read_file(img_path), channels=3)
    elif ext.lower() == 'jpg':
        img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    else:
        img = None
        print('cannot process {} file.'.format(ext.lower()))

    return img, filename


def preprocess(img, h, w):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
    pad_img = tf.expand_dims(pad_img, dim=0)

    return pad_img


def decode_labels(mask, img_shape, num_classes):
    color_table = palette

    color_mat = tf.constant(color_table, dtype=tf.float32)
    print('debug', color_mat)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))

    return pred


def main():
    args = get_arguments()

    # load parameters
    if args.dataset != 'voc2012':
        print('Invalid dataset {}'.format(args.dataset))
        return
    else:
        param = VOC2012_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    args.img_path = '/home/melody/data/VOCdevkit/VOC2012/JPEGImages/2007_003201.jpg'
    # preprocess images
    img, filename = load_img(args.img_path)
    img_shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
    img = preprocess(img, h, w)

    # Create network.
    net = PSPNet({'data': img}, trainable=False)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, trainable=False)

    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = decode_labels(raw_output_up, img_shape, num_classes)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()

    net.load('/home/melody/develop/caffe-tensorflow/pspnet.npy', sess)

    saver = tf.train.Saver()

    save(saver, sess, SAVE_DIR, 0)

    preds = sess.run(pred)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    misc.imsave(args.save_dir + filename, preds[0])


if __name__ == '__main__':
    main()
