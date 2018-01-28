from kaffe.tensorflow import Network
import tensorflow as tf


class pspnet101_VOC2012(Network):
    def setup(self, is_training):
        (self.feed('data')
         .conv(3, 3, 64, 2, 2, padding='SAME', biased=False, relu=False, name='conv1_1_3x3_s2')
         .batch_normalization(relu=True, name='conv1_1_3x3_s2_bn')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv1_2_3x3')
         .batch_normalization(relu=True, name='conv1_2_3x3_bn')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv1_3_3x3')
         .batch_normalization(relu=True, name='conv1_3_3x3_bn')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
         .conv(1, 1, 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_1_3x3')
         .batch_normalization(relu=True, name='conv2_1_3x3_bn')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_1_1x1_increase')
         .batch_normalization(name='conv2_1_1x1_increase_bn'))

        (self.feed('pool1_3x3_s2')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_1_1x1_proj')
         .batch_normalization(name='conv2_1_1x1_proj_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
         .add(name='conv2_1')
         .relu(name='conv2_1_relu')
         .conv(1, 1, 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_2_3x3')
         .batch_normalization(relu=True, name='conv2_2_3x3_bn')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_2_1x1_increase')
         .batch_normalization(name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1_relu',
                   'conv2_2_1x1_increase_bn')
         .add(name='conv2_2')
         .relu(name='conv2_2_relu')
         .conv(1, 1, 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_3_3x3')
         .batch_normalization(relu=True, name='conv2_3_3x3_bn')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv2_3_1x1_increase')
         .batch_normalization(name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2_relu',
                   'conv2_3_1x1_increase_bn')
         .add(name='conv2_3')
         .relu(name='conv2_3_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
         .conv(3, 3, 128, 2, 2, padding='SAME', biased=False, relu=False, name='conv3_1_3x3')
         .batch_normalization(relu=True, name='conv3_1_3x3_bn')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_1_1x1_increase')
         .batch_normalization(name='conv3_1_1x1_increase_bn'))

        (self.feed('conv2_3_relu')
         .conv(1, 1, 512, 2, 2, padding='SAME', biased=False, relu=False, name='conv3_1_1x1_proj')
         .batch_normalization(name='conv3_1_1x1_proj_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
         .add(name='conv3_1')
         .relu(name='conv3_1_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_2_3x3')
         .batch_normalization(relu=True, name='conv3_2_3x3_bn')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_2_1x1_increase')
         .batch_normalization(name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1_relu',
                   'conv3_2_1x1_increase_bn')
         .add(name='conv3_2')
         .relu(name='conv3_2_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_3_3x3')
         .batch_normalization(relu=True, name='conv3_3_3x3_bn')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_3_1x1_increase')
         .batch_normalization(name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2_relu',
                   'conv3_3_1x1_increase_bn')
         .add(name='conv3_3')
         .relu(name='conv3_3_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_4_1x1_reduce')
         .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_4_3x3')
         .batch_normalization(relu=True, name='conv3_4_3x3_bn')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv3_4_1x1_increase')
         .batch_normalization(name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3_relu',
                   'conv3_4_1x1_increase_bn')
         .add(name='conv3_4')
         .relu(name='conv3_4_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_1_3x3')
         .batch_normalization(relu=True, name='conv4_1_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_1_1x1_increase')
         .batch_normalization(name='conv4_1_1x1_increase_bn'))

        (self.feed('conv3_4_relu')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_1_1x1_proj')
         .batch_normalization(name='conv4_1_1x1_proj_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
         .add(name='conv4_1')
         .relu(name='conv4_1_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_2_3x3')
         .batch_normalization(relu=True, name='conv4_2_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_2_1x1_increase')
         .batch_normalization(name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1_relu',
                   'conv4_2_1x1_increase_bn')
         .add(name='conv4_2')
         .relu(name='conv4_2_relu')
         .conv(1, 1, 21, 1, 1, padding='SAME', relu=False, name='conv_aux')
         # .interp(None, None, 21, 8, 1, 0, 0, name='conv_aux_interp')
         )

        (self.feed('conv4_2_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_3_3x3')
         .batch_normalization(relu=True, name='conv4_3_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_3_1x1_increase')
         .batch_normalization(name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2_relu',
                   'conv4_3_1x1_increase_bn')
         .add(name='conv4_3')
         .relu(name='conv4_3_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_4_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_4_3x3')
         .batch_normalization(relu=True, name='conv4_4_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_4_1x1_increase')
         .batch_normalization(name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3_relu',
                   'conv4_4_1x1_increase_bn')
         .add(name='conv4_4')
         .relu(name='conv4_4_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_5_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_5_3x3')
         .batch_normalization(relu=True, name='conv4_5_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_5_1x1_increase')
         .batch_normalization(name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4_relu',
                   'conv4_5_1x1_increase_bn')
         .add(name='conv4_5')
         .relu(name='conv4_5_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_6_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_6_3x3')
         .batch_normalization(relu=True, name='conv4_6_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_6_1x1_increase')
         .batch_normalization(name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5_relu',
                   'conv4_6_1x1_increase_bn')
         .add(name='conv4_6')
         .relu(name='conv4_6_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_7_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_7_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_7_3x3')
         .batch_normalization(relu=True, name='conv4_7_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_7_1x1_increase')
         .batch_normalization(name='conv4_7_1x1_increase_bn'))

        (self.feed('conv4_6_relu',
                   'conv4_7_1x1_increase_bn')
         .add(name='conv4_7')
         .relu(name='conv4_7_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_8_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_8_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_8_3x3')
         .batch_normalization(relu=True, name='conv4_8_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_8_1x1_increase')
         .batch_normalization(name='conv4_8_1x1_increase_bn'))

        (self.feed('conv4_7_relu',
                   'conv4_8_1x1_increase_bn')
         .add(name='conv4_8')
         .relu(name='conv4_8_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_9_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_9_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_9_3x3')
         .batch_normalization(relu=True, name='conv4_9_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_9_1x1_increase')
         .batch_normalization(name='conv4_9_1x1_increase_bn'))

        (self.feed('conv4_8_relu',
                   'conv4_9_1x1_increase_bn')
         .add(name='conv4_9')
         .relu(name='conv4_9_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_10_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_10_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_10_3x3')
         .batch_normalization(relu=True, name='conv4_10_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_10_1x1_increase')
         .batch_normalization(name='conv4_10_1x1_increase_bn'))

        (self.feed('conv4_9_relu',
                   'conv4_10_1x1_increase_bn')
         .add(name='conv4_10')
         .relu(name='conv4_10_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_11_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_11_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_11_3x3')
         .batch_normalization(relu=True, name='conv4_11_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_11_1x1_increase')
         .batch_normalization(name='conv4_11_1x1_increase_bn'))

        (self.feed('conv4_10_relu',
                   'conv4_11_1x1_increase_bn')
         .add(name='conv4_11')
         .relu(name='conv4_11_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_12_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_12_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_12_3x3')
         .batch_normalization(relu=True, name='conv4_12_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_12_1x1_increase')
         .batch_normalization(name='conv4_12_1x1_increase_bn'))

        (self.feed('conv4_11_relu',
                   'conv4_12_1x1_increase_bn')
         .add(name='conv4_12')
         .relu(name='conv4_12_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_13_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_13_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_13_3x3')
         .batch_normalization(relu=True, name='conv4_13_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_13_1x1_increase')
         .batch_normalization(name='conv4_13_1x1_increase_bn'))

        (self.feed('conv4_12_relu',
                   'conv4_13_1x1_increase_bn')
         .add(name='conv4_13')
         .relu(name='conv4_13_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_14_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_14_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_14_3x3')
         .batch_normalization(relu=True, name='conv4_14_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_14_1x1_increase')
         .batch_normalization(name='conv4_14_1x1_increase_bn'))

        (self.feed('conv4_13_relu',
                   'conv4_14_1x1_increase_bn')
         .add(name='conv4_14')
         .relu(name='conv4_14_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_15_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_15_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_15_3x3')
         .batch_normalization(relu=True, name='conv4_15_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_15_1x1_increase')
         .batch_normalization(name='conv4_15_1x1_increase_bn'))

        (self.feed('conv4_14_relu',
                   'conv4_15_1x1_increase_bn')
         .add(name='conv4_15')
         .relu(name='conv4_15_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_16_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_16_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_16_3x3')
         .batch_normalization(relu=True, name='conv4_16_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_16_1x1_increase')
         .batch_normalization(name='conv4_16_1x1_increase_bn'))

        (self.feed('conv4_15_relu',
                   'conv4_16_1x1_increase_bn')
         .add(name='conv4_16')
         .relu(name='conv4_16_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_17_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_17_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_17_3x3')
         .batch_normalization(relu=True, name='conv4_17_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_17_1x1_increase')
         .batch_normalization(name='conv4_17_1x1_increase_bn'))

        (self.feed('conv4_16_relu',
                   'conv4_17_1x1_increase_bn')
         .add(name='conv4_17')
         .relu(name='conv4_17_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_18_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_18_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_18_3x3')
         .batch_normalization(relu=True, name='conv4_18_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_18_1x1_increase')
         .batch_normalization(name='conv4_18_1x1_increase_bn'))

        (self.feed('conv4_17_relu',
                   'conv4_18_1x1_increase_bn')
         .add(name='conv4_18')
         .relu(name='conv4_18_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_19_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_19_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_19_3x3')
         .batch_normalization(relu=True, name='conv4_19_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_19_1x1_increase')
         .batch_normalization(name='conv4_19_1x1_increase_bn'))

        (self.feed('conv4_18_relu',
                   'conv4_19_1x1_increase_bn')
         .add(name='conv4_19')
         .relu(name='conv4_19_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_20_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_20_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_20_3x3')
         .batch_normalization(relu=True, name='conv4_20_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_20_1x1_increase')
         .batch_normalization(name='conv4_20_1x1_increase_bn'))

        (self.feed('conv4_19_relu',
                   'conv4_20_1x1_increase_bn')
         .add(name='conv4_20')
         .relu(name='conv4_20_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_21_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_21_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_21_3x3')
         .batch_normalization(relu=True, name='conv4_21_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_21_1x1_increase')
         .batch_normalization(name='conv4_21_1x1_increase_bn'))

        (self.feed('conv4_20_relu',
                   'conv4_21_1x1_increase_bn')
         .add(name='conv4_21')
         .relu(name='conv4_21_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_22_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_22_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_22_3x3')
         .batch_normalization(relu=True, name='conv4_22_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_22_1x1_increase')
         .batch_normalization(name='conv4_22_1x1_increase_bn'))

        (self.feed('conv4_21_relu',
                   'conv4_22_1x1_increase_bn')
         .add(name='conv4_22')
         .relu(name='conv4_22_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_23_1x1_reduce')
         .batch_normalization(relu=True, name='conv4_23_1x1_reduce_bn')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='conv4_23_3x3')
         .batch_normalization(relu=True, name='conv4_23_3x3_bn')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False, name='conv4_23_1x1_increase')
         .batch_normalization(name='conv4_23_1x1_increase_bn'))

        (self.feed('conv4_22_relu',
                   'conv4_23_1x1_increase_bn')
         .add(name='conv4_23')
         .relu(name='conv4_23_relu')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_1_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='conv5_1_3x3')
         .batch_normalization(relu=True, name='conv5_1_3x3_bn')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_1_1x1_increase')
         .batch_normalization(name='conv5_1_1x1_increase_bn'))

        (self.feed('conv4_23_relu')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_1_1x1_proj')
         .batch_normalization(name='conv5_1_1x1_proj_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
         .add(name='conv5_1')
         .relu(name='conv5_1_relu')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_2_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='conv5_2_3x3')
         .batch_normalization(relu=True, name='conv5_2_3x3_bn')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_2_1x1_increase')
         .batch_normalization(name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1_relu',
                   'conv5_2_1x1_increase_bn')
         .add(name='conv5_2')
         .relu(name='conv5_2_relu')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_3_1x1_reduce')
         .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='conv5_3_3x3')
         .batch_normalization(relu=True, name='conv5_3_3x3_bn')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_3_1x1_increase')
         .batch_normalization(name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2_relu',
                   'conv5_3_1x1_increase_bn')
         .add(name='conv5_3')
         .relu(name='conv5_3_relu'))

        conv5_3 = self.layers['conv5_3_relu']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('conv5_3_relu')
         .avg_pool(60, 60, 60, 60, padding='SAME', name='conv5_3_pool1')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_3_pool1_conv')
         .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool1_interp')
         .interp(shape, name='conv5_3_pool1_interp')
         )

        (self.feed('conv5_3_relu')
         .avg_pool(30, 30, 30, 30, padding='SAME', name='conv5_3_pool2')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_3_pool2_conv')
         .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool2_interp')
         .interp(shape, name='conv5_3_pool2_interp')
         )

        (self.feed('conv5_3_relu')
         .avg_pool(20, 20, 20, 20, padding='SAME', name='conv5_3_pool3')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_3_pool3_conv')
         .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool3_interp')
         .interp(shape, name='conv5_3_pool3_interp')
         )

        (self.feed('conv5_3_relu')
         .avg_pool(10, 10, 10, 10, padding='SAME', name='conv5_3_pool6')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_3_pool6_conv')
         .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool6_interp')
         .interp(shape, name='conv5_3_pool6_interp')
         )

        (self.feed('conv5_3_relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
         .concat(3, name='conv5_3_concat')
         .conv(3, 3, 512, 1, 1, padding='SAME', biased=False, relu=False, name='conv5_4')
         .batch_normalization(relu=True, name='conv5_4_bn')
         .conv(1, 1, 21, 1, 1, padding='SAME', relu=False, name='conv6')
         # .interp(None, None, 21, 8, 1, 0, 0, name='conv6_interp')
         )
