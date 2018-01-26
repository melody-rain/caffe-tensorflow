# -*- coding: utf-8 -*-
import argparse
import json
import os
import numpy as np
from numpy import newaxis
from matplotlib import pyplot as plt
import shutil
import time
import sys
import skimage.io
from PIL import Image
from logger import logger
from collections import Counter
import operator
import config
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import os.path as osp

import sys
sys.path.insert(0, '/home/melody/hdd1/develop/caffe-cuhk/python')
import caffe


def convert_img(pil_img, net_name='ssd'):
    if net_name == 'ssd':
        img = pil_img.resize((300, 300), Image.BILINEAR)
    else:
        img = pil_img.resize((224, 224), Image.BILINEAR)
    img = np.array(img)
    img = skimage.img_as_float(img).astype(np.float32)

    if img.ndim == 2:
        img_temp = np.zeros(shape=(img.shape[0], img.shape[1], 3))
        img_temp[:, :, 0] = img
        img_temp[:, :, 1] = img
        img_temp[:, :, 2] = img
        img = img_temp
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


class PartClassifier:
    def __init__(self, _model_name, _deploy_prototxt, type, num_class,
                 cpu_mode=False, debug=False, gpu_id=0):

        self.model_name = _model_name
        self.deploy_prototxt = _deploy_prototxt

        self.gpu_id = gpu_id
        self.type = type
        self.num_class = num_class
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)

        self.net = caffe.Net(_deploy_prototxt, _model_name, caffe.TEST)
        if self.net is not None:
            logger.info('{} successfully loaded'.format(_model_name))

        self.debug = debug

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))

        self.transformer.set_mean('data', np.asarray([103.94,116.78,123.68]))

        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_input_scale('data', 0.017)
        self.transformer.set_channel_swap('data', (2, 1, 0))

    @property
    def gpu_id(self):
        return self.gpu_id

    @property
    def type(self):
        return self.type

    def isPNG(self, filename):
        dot = filename.rfind(".")
        if dot != 0:
            file_format = filename[dot + 1:]
            if file_format in ["png", "PNG"]:
                return True

        return False

    def predict_impl(self, caffe_img, NLOG = False):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe_img)
        out = self.net.forward()
        prediction = out['prob'][0]
        rlt = prediction.argmax()

        str_rlt = config.index2label[rlt]
        if NLOG:
            logger.info('Predicted: ' + str_rlt + ' with confidence: ' + str(prediction[rlt]))

        return str_rlt, prediction[rlt]

    # predict with filename
    def predict1(self, full_filename, NLOG = False):
        file_stat = os.stat(full_filename)
        if file_stat.st_size == 0:
            logger.info('Invalid image: ' + full_filename)

        return self.predict_impl(caffe.io.load_image(full_filename), NLOG = NLOG)

    # predict with image array
    def predict2(self, image_array, NLOG = False):
        img = skimage.img_as_float(image_array).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.shape[2] == 4:
            img = img[:, :, 3]

        return self.predict_impl(img, NLOG = NLOG)

    def predict4(self, caffe_img):
        caffe.set_device(self.gpu_id)
        self.net.blobs['data'].reshape(caffe_img.shape[0], caffe_img.shape[3], caffe_img.shape[1], caffe_img.shape[2])
        for i in xrange(caffe_img.shape[0]):
            self.net.blobs['data'].data[i, ...] = self.transformer.preprocess('data', caffe_img[i, ...])
        out = self.net.forward()
        out = out['prob'].squeeze()
        out_shape = out.shape
        if len(out_shape) == 1:
            return out.reshape(-1, out_shape[0])
        return out

    def predict3(self, caffe_img, NLOG = False):
        prediction = self.predict4(caffe_img)
        rlt = prediction.argmax(axis=1)

        str_rlt = [config.index2label[a_res] for a_res in rlt]
        predictions = [prediction[i][a_res] for i, a_res in enumerate(rlt)]
        if NLOG:
            for i, a_res in enumerate(rlt):
                logger.info('Predicted: ' + config.index2label[a_res] + ' with confidence: ' + str(predictions[i]))

        return str_rlt, predictions

    def accuracy(self, caffe_img, labels, NLOG=False):
        str_rlt, _ = self.predict3(caffe_img, NLOG=NLOG)
        assert len(str_rlt) == len(labels), \
            'number of labels[{}] should be equal to number of output[{}]'.format(len(str_rlt), len(labels))

        correct = 0
        for i, a_res in enumerate(str_rlt):
            if labels[i] == a_res:
                correct += 1

        accuracy = 1.0 * correct / len(labels)
        logger.info('Accuracy for the batch: {0:02f}'.format(accuracy))
        return correct

    def predict_batch(self, dataPath, target_class, out_data_path=None):
        file_cnt = 0

        if out_data_path is not None:
            if os.path.exists(out_data_path):
                shutil.rmtree(out_data_path)

                os.mkdir(out_data_path)
        total_files = len(os.listdir(dataPath))

        for root, dirs, files in os.walk(dataPath):
            for file in files:
                file_full_path = os.path.join(root, file)

                file_stat = os.stat(file_full_path)
                if file_stat.st_size == 0:
                    continue

                self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(file_full_path))
                out = self.net.forward()
                prediction = out['prob'][0]
                rlt = prediction.argmax()

                rlt_chi = config.index2label[rlt]
                if rlt_chi == target_class:
                    new_filename = str(file_cnt) + ".png"
                    # shutil.copy(os.path.join(root, file), os.path.join(out_data_path, file))
                    file_cnt += 1

        logger.info(('{:.2f}{}{}').format(100.0 * file_cnt / total_files, '%\tbelongs to\t', target_class))


class FusedModel:
    def __init__(self, config_file, num_class, etc_path, model_path):
        with open(os.path.join(etc_path, config_file)) as fd:
            configs = json.load(fd)

        model_info = configs['model']
        self.data_type = []
        self.weights = []
        self.classifiers = []
        self.num_class = num_class
        for a_model in model_info:
            self.classifiers.append(
                PartClassifier(os.path.join(model_path, str(a_model['model_file'])),
                                 os.path.join(etc_path, str(a_model['deploy_file'])),
                                 str(a_model['type']),
                                 num_class,
                                 gpu_id=a_model['gpu_id']))

            if a_model['type'] not in self.data_type:
                self.data_type.append(a_model['type'])

            assert len(a_model['weights']) == num_class, '{} vs {}'.format(len(a_model['weights']), num_class)
            w = []
            for i in xrange(num_class):
                w.append(a_model['weights'][str(i)])

            s = sum(w)
            w = np.asarray(w) / s
            w = list(w)
            self.weights.append(w)

        if len(self.weights) == 1:
            self.weights[0] = [1] * num_class

        for i, a_model in enumerate(model_info):
            logger.info('Model: {}'.format(a_model['type']))
            logger.info(self.weights[i])

    def predict1(self, im_name):
        datas = {}
        for a_data_type in self.data_type:
            datas[a_data_type] = []

        img = Image.open(im_name)
        for a_data_type in self.data_type:
            datas[a_data_type].append(convert_img(img, net_name=str(a_data_type)))

        for a_data_type in self.data_type:
            datas[a_data_type] = np.asarray(datas[a_data_type])

        pred = np.zeros((1, self.num_class))

        for i, a_classifier in enumerate(self.classifiers):
                pred += np.asarray(self.weights[i]) \
                                                  * a_classifier.predict4(datas[a_classifier.type])

        rlt = pred.argmax(axis=1)
        logger.info('Predicted as: {} with conf: {}'.format(config.index2label[rlt[0]], pred[0][rlt[0]]))
        return pred

    def predict_file_list(self, file_list, batch_size, save_path=''):

        pred = np.zeros((len(file_list), self.num_class))

        global_cnt = 0
        for ii in xrange(0, len(file_list), batch_size):
            datas = {}
            for a_data_type in self.data_type:
                datas[a_data_type] = []
            img_index = []

            for index, im_name in enumerate(file_list[ii: ii + batch_size]):
                try:
                    img = Image.open(im_name)
                except Exception as e:
                    logger.info(e)
                    logger.info('Create an empty image')
                    img = np.zeros((300, 300, 3), dtype=np.uint8)
                    img = Image.fromarray(img)

                for a_data_type in self.data_type:
                    datas[a_data_type].append(convert_img(img, net_name=str(a_data_type)))
                img_index.append(index + ii)

            for a_data_type in self.data_type:
                datas[a_data_type] = np.asarray(datas[a_data_type])

            for i, a_classifier in enumerate(self.classifiers):
                pred[ii: ii + batch_size, ...] += np.asarray(self.weights[i]) \
                                                  * a_classifier.predict4(datas[a_classifier.type])

                pred_class = pred[ii : ii + batch_size, ...].argmax(axis=1)

                if save_path != '':
                    for index_image, a_pred_cls in enumerate(pred_class.tolist()):
                        target_path = os.path.join(save_path, '{}'.format(a_pred_cls))
                        if not os.path.exists(target_path):
                            os.mkdir(target_path)
                        if global_cnt % 1000 == 0:
                            print 'Finished {0:02f}%'.format(global_cnt * 1.0 / len(file_list) * 100)
                        shutil.copy(file_list[ii + index_image], target_path)
                        global_cnt += 1
        return pred

    def predict(self, data_src, batch_size, save_res=True):
        if not os.path.isdir(data_src):
            return None

        if save_res:
            save_path = '{}_res'.format(data_src)
            if os.path.exists(save_path):
               shutil.rmtree(save_path)
            os.mkdir(save_path)
        else:
            save_path = ''

        file_list = os.listdir(data_src)
        file_list_ = [os.path.join(data_src, a_file) for a_file in file_list if a_file.endswith('.jpg')]

        return self.predict_file_list(file_list_, batch_size, save_path=save_path)

    def to_index(self, target):
        try:
            a_cls = int(target)
        except ValueError:
            a_cls = int(config.label2index[target])

        return a_cls


    def predict_acc(self, data_src, batch_size, save_wrong=False, with_vote=False):
        classes = os.listdir(data_src)

        corrects = [0] * self.num_class
        totals = [0] * self.num_class
        hits = [0] * self.num_class

        for a_class in classes:
            temp_path = os.path.join(data_src, a_class)
            if save_wrong:
                NC_dir = temp_path + '_NC'
                if not os.path.exists(NC_dir):
                    os.mkdir(NC_dir)

            if os.path.isdir(temp_path):
                n_correct = 0
                a_class = self.to_index(a_class)
                file_list = os.listdir(temp_path)
                labels = [a_class] * len(file_list)
                file_list_ = [os.path.join(temp_path, a_file) for a_file in file_list]

                totals[int(a_class)] = len(file_list_)
                for ii in xrange(0, len(file_list_), batch_size):
                    datas = {}
                    for a_data_type in self.data_type:
                        datas[a_data_type] = []
                    img_index = []
                    for index, im_name in enumerate(file_list_[ii: ii + batch_size]):
                        img = Image.open(im_name)
                        for a_data_type in self.data_type:
                            datas[a_data_type].append(convert_img(img, net_name=str(a_data_type)))
                        img_index.append(index + ii)

                    for a_data_type in self.data_type:
                        datas[a_data_type] = np.asarray(datas[a_data_type])

                    if (ii + args.batch_size) >= len(file_list):
                        if with_vote:
                            pred = np.zeros((len(file_list) - ii, len(self.classifiers)))
                        else:
                            pred = np.zeros((len(file_list) - ii, self.num_class))
                    else:
                        if with_vote:
                            pred = np.zeros((args.batch_size, len(self.classifiers)))
                        else:
                            pred = np.zeros((args.batch_size, self.num_class))

                    for i, a_classifier in enumerate(self.classifiers):
                        temp = a_classifier.predict4(datas[a_classifier.type])
                        if with_vote:
                            pred[:, i] = temp.argmax(axis=1)
                        else:
                            pred += np.asarray(self.weights[i]) * temp

                    if with_vote:
                        rlt = []
                        for i in xrange(0, pred.shape[0]):
                            elements, counts = np.unique(pred[i, :], return_counts=True)
                            rlt.append(elements[counts.argmax()])
                    else:
                        rlt = pred.argmax(axis=1)

                    for i, a_res in enumerate(rlt):
                        if save_wrong and a_res != a_class:
                            shutil.copy(file_list_[i + ii], NC_dir)

                        hits[self.to_index(a_res)] += 1
                        if labels[ii + i] == a_res:
                            n_correct += 1
                            corrects[int(a_res)] += 1

                accuracy = 1.0 * n_correct / len(labels)
                logger.info('Recall for class {} : {}'.format(a_class, accuracy))

        logger.info(totals)
        logger.info(corrects)
        logger.info(hits)
        logger.info('Precision: ')
        logger.info(1.0 * np.asarray(corrects) / np.asarray(hits))
        logger.info('Recall: ')
        logger.info(1.0 * np.asarray(corrects) / np.asarray(totals))


def parse_class(file_name):
    parts = file_name.split('/')
    return parts[-2]


def assemble_predicted_res(predicted, img_list, final_path):
    if final_path != '':
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.mkdir(final_path)

    _part_cls = {}
    predicted_cls = predicted.argmax(axis=1)
    conf = [predicted[i][c] for i, c in enumerate(list(predicted_cls))]

    assert len(img_list) == predicted.shape[0], '{} vs {}'.format(len(img_list), predicted.shape[0])

    for i, a_img in enumerate(img_list):
        cls = parse_class(a_img)
        cls_index = int(config.label2index[cls])
        if (cls_index == predicted_cls[i]):# or (conf[i] >= 0.8):  # FIXME
            if _part_cls.has_key(predicted_cls[i]):
                _part_cls[predicted_cls[i]].append((a_img, conf[i]))
            else:
                _part_cls[predicted_cls[i]] = [(a_img, conf[i])]

            # copy or delete the image
            if final_path != '':
                temp_path = os.path.join(final_path, cls)
                if not os.path.exists(temp_path):
                    os.mkdir(temp_path)
                shutil.copy(a_img, os.path.join(temp_path))
        elif final_path == '':
            os.remove(a_img)

    return _part_cls


# type: interface
# description: classify images into car parts
# target_folder: path contains cropped parts from part locator
def classify_part(target_folder, fused_model, final_path='',
                  th=0.0, batch_size=32, save_res=True):
    im_names = []
    valid_format = ['jpg', 'JPG', 'png']

    sub_files = os.listdir(target_folder)
    part_folders = []
    for a_subfile in sub_files:
        if os.path.isdir(os.path.join(target_folder, a_subfile)):
            part_folders.append(os.path.join(target_folder, a_subfile))

    for a_folder in part_folders:
        for root, _, files in os.walk(a_folder):
            for a_file in files:
                dot_pos = a_file.rfind('.')
                file_extension = a_file[dot_pos + 1:]

                if file_extension not in valid_format:
                    logger.info('Ignore ' + os.path.join(root, a_file))
                    # ignore those files for now
                    continue
                im_names.append(os.path.join(root, a_file))

    pred = fused_model.predict_file_list(im_names, batch_size)

    _part_cls = assemble_predicted_res(pred, im_names, final_path)

    for k, v in _part_cls.iteritems():
        new_v = []
        for i, a_v in enumerate(v):
            if a_v[1] >= th:
                new_v.append(a_v)
        _part_cls[k] = new_v

    if save_res:
        file_name = os.path.basename(target_folder)
        with open(os.path.join(target_folder, 'predict_part_result_{}.txt'.format(file_name)), 'w') as res_fd:
            content = '{:25s}{:11s}\t{:s}\n'.format('part name', 'predicted part', 'confidence')
            res_fd.write(content)

            for k, v in _part_cls.iteritems():
                for i, a_v in enumerate(v):
                    content = '{:25s}{:s}\t{:.3f}\n'.format(config.index2label[k], a_v[0], a_v[1])
                    res_fd.write(content)

    return _part_cls


def predict(data_source, args):
    fused_model = FusedModel('part_{}_config.json'.format(args.num_class),
                             args.num_class, args.etc_path, args.model_path)
    print fused_model.predict(data_source, args.batch_size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classify vehicle parts')
    parser.add_argument('--test-type', dest='test_type', help='test type. \
                        0 for single image test. 1 for batch test. 2 for batch test with accuracy', \
                        default=0, type=int)
    parser.add_argument('--data', dest='data_source', help='The directory of data to predict.', type=str, default='')
    parser.add_argument('--batch-size', dest='batch_size', help='batch size of the part classifier', type=int, default=32)
    parser.add_argument('--save-wrong', dest='save_wrong', help='Save incorrect results', action='store_true')
    parser.add_argument('--vote', dest='with_vote', help='Save incorrect results', action='store_true')
    parser.add_argument('--num-class', dest='num_class', help='number of classes of the network', type=int, required=True)
    parser.add_argument('--etc-path', dest='etc_path', help='path to the etc files', type=str, default='')
    parser.add_argument('--model-path', dest='model_path', help='path to the etc files', type=str, default='')

    args = parser.parse_args()
    batch_size = args.batch_size

    if args.etc_path == '':
        args.etc_path = os.path.join(osp.dirname(__file__), 'etc')

    if args.model_path == '':
        args.model_path = os.path.join(osp.dirname(__file__),
                                       'etc')

    if args.test_type == 0:
        fused_model = FusedModel('class_{}_config.json'.format(args.num_class),
                                 args.num_class, args.etc_path, args.model_path)

        while True:
            file_full_name = raw_input('Input the image: ')
            fused_model.predict1(file_full_name)
    elif args.test_type == 1:
        if os.path.exists(args.data_source):
            fused_model = FusedModel('class_{}_config.json'.format(args.num_class),
                                     args.num_class, args.etc_path, args.model_path)
            fused_model.predict(args.data_source, args.batch_size)
        else:
            while True:
                data_source = raw_input('Input the data source: ')
                predict(data_source, args)
    elif args.test_type == 2:
        fused_model = FusedModel('class_{}_config.json'.format(args.num_class),
                                 args.num_class, args.etc_path, args.model_path)
        fused_model.predict_acc(args.data_source, args.batch_size, save_wrong=args.save_wrong, with_vote=args.with_vote)
    else:
        logger.info('Unsupported test. Exit.')