from kaffe.tensorflow import Network
import tensorflow as tf


class pspnet101_VOC2012(Network):
    def setup(self, is_training):
        (self.feed('data')
         .conv(3, 3, 64, 2, 2, padding='SAME', biased=False, relu=False, name='pspnet_v1_101/root/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/root/conv1/BatchNorm')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False, name='pspnet_v1_101/root/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/root/conv2/BatchNorm')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False, name='pspnet_v1_101/root/conv3')
         .batch_normalization(relu=True, name='pspnet_v1_101/root/conv3/BatchNorm')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
         .conv(1, 1, 64, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_1/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block1/unit_1/bottleneck_v1/conv1/BatchNorm')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_1/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block1/unit_1/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_1/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block1/unit_1/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('pool1_3x3_s2')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_1/bottleneck_v1/shortcut')
         .batch_normalization(name='pspnet_v1_101/block1/unit_1/bottleneck_v1/shortcut/BatchNorm'))

        (self.feed('pspnet_v1_101/block1/unit_1/bottleneck_v1/shortcut/BatchNorm',
                   'pspnet_v1_101/block1/unit_1/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv2_1')
         .relu(name='conv2_1_relu')
         .conv(1, 1, 64, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_2/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block1/unit_2/bottleneck_v1/conv1/BatchNorm')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_2/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block1/unit_2/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_2/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block1/unit_2/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv2_1_relu',
                   'pspnet_v1_101/block1/unit_2/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv2_2')
         .relu(name='conv2_2_relu')
         .conv(1, 1, 64, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_3/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block1/unit_3/bottleneck_v1/conv1/BatchNorm')
         .conv(3, 3, 64, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_3/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block1/unit_3/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block1/unit_3/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block1/unit_3/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv2_2_relu',
                   'pspnet_v1_101/block1/unit_3/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv2_3')
         .relu(name='conv2_3_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_1/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_1/bottleneck_v1/conv1/BatchNorm')
         .conv(3, 3, 128, 2, 2, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_1/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_1/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_1/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block2/unit_1/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv2_3_relu')
         .conv(1, 1, 512, 2, 2, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_1/bottleneck_v1/shortcut')
         .batch_normalization(name='pspnet_v1_101/block2/unit_1/bottleneck_v1/shortcut/BatchNorm'))

        (self.feed('pspnet_v1_101/block2/unit_1/bottleneck_v1/shortcut/BatchNorm',
                   'pspnet_v1_101/block2/unit_1/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv3_1')
         .relu(name='conv3_1_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_2/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_2/bottleneck_v1/conv1/BatchNorm')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_2/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_2/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_2/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block2/unit_2/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv3_1_relu',
                   'pspnet_v1_101/block2/unit_2/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv3_2')
         .relu(name='conv3_2_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_3/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_3/bottleneck_v1/conv1/BatchNorm')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_3/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_3/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_3/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block2/unit_3/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv3_2_relu',
                   'pspnet_v1_101/block2/unit_3/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv3_3')
         .relu(name='conv3_3_relu')
         .conv(1, 1, 128, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_4/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_4/bottleneck_v1/conv1/BatchNorm')
         .conv(3, 3, 128, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_4/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block2/unit_4/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block2/unit_4/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block2/unit_4/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv3_3_relu',
                   'pspnet_v1_101/block2/unit_4/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv3_4')
         .relu(name='conv3_4_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_1/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_1/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_1/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_1/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_1/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_1/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv3_4_relu')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_1/bottleneck_v1/shortcut')
         .batch_normalization(name='pspnet_v1_101/block3/unit_1/bottleneck_v1/shortcut/BatchNorm'))

        (self.feed('pspnet_v1_101/block3/unit_1/bottleneck_v1/shortcut/BatchNorm',
                   'pspnet_v1_101/block3/unit_1/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_1')
         .relu(name='conv4_1_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_2/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_2/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_2/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_2/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_2/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_2/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_1_relu',
                   'pspnet_v1_101/block3/unit_2/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_2')
         .relu(name='conv4_2_relu')
         .conv(1, 1, 2, 1, 1, padding='SAME', relu=False, name='pspnet_v1_101/aux_logits')
         # .interp(None, None, 21, 8, 1, 0, 0, name='conv_aux_interp')
         )

        (self.feed('conv4_2_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_3/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_3/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_3/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_3/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_3/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_3/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_2_relu',
                   'pspnet_v1_101/block3/unit_3/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_3')
         .relu(name='conv4_3_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_4/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_4/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_4/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_4/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_4/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_4/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_3_relu',
                   'pspnet_v1_101/block3/unit_4/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_4')
         .relu(name='conv4_4_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_5/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_5/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_5/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_5/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_5/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_5/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_4_relu',
                   'pspnet_v1_101/block3/unit_5/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_5')
         .relu(name='conv4_5_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_6/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_6/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_6/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_6/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_6/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_6/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_5_relu',
                   'pspnet_v1_101/block3/unit_6/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_6')
         .relu(name='conv4_6_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_7/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_7/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_7/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_7/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_7/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_7/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_6_relu',
                   'pspnet_v1_101/block3/unit_7/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_7')
         .relu(name='conv4_7_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_8/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_8/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_8/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_8/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_8/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_8/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_7_relu',
                   'pspnet_v1_101/block3/unit_8/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_8')
         .relu(name='conv4_8_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_9/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_9/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_9/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_9/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_9/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_9/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_8_relu',
                   'pspnet_v1_101/block3/unit_9/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_9')
         .relu(name='conv4_9_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_10/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_10/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_10/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_10/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_10/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_10/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_9_relu',
                   'pspnet_v1_101/block3/unit_10/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_10')
         .relu(name='conv4_10_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_11/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_11/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_11/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_11/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_11/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_11/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_10_relu',
                   'pspnet_v1_101/block3/unit_11/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_11')
         .relu(name='conv4_11_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_12/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_12/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_12/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_12/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_12/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_12/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_11_relu',
                   'pspnet_v1_101/block3/unit_12/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_12')
         .relu(name='conv4_12_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_13/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_13/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_13/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_13/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_13/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_13/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_12_relu',
                   'pspnet_v1_101/block3/unit_13/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_13')
         .relu(name='conv4_13_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_14/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_14/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_14/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_14/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_14/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_14/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_13_relu',
                   'pspnet_v1_101/block3/unit_14/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_14')
         .relu(name='conv4_14_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_15/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_15/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_15/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_15/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_15/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_15/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_14_relu',
                   'pspnet_v1_101/block3/unit_15/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_15')
         .relu(name='conv4_15_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_16/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_16/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_16/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_16/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_16/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_16/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_15_relu',
                   'pspnet_v1_101/block3/unit_16/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_16')
         .relu(name='conv4_16_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_17/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_17/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_17/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_17/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_17/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_17/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_16_relu',
                   'pspnet_v1_101/block3/unit_17/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_17')
         .relu(name='conv4_17_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_18/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_18/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_18/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_18/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_18/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_18/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_17_relu',
                   'pspnet_v1_101/block3/unit_18/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_18')
         .relu(name='conv4_18_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_19/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_19/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_19/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_19/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_19/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_19/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_18_relu',
                   'pspnet_v1_101/block3/unit_19/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_19')
         .relu(name='conv4_19_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_20/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_20/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_20/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_20/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_20/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_20/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_19_relu',
                   'pspnet_v1_101/block3/unit_20/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_20')
         .relu(name='conv4_20_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_21/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_21/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_21/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_21/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_21/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_21/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_20_relu',
                   'pspnet_v1_101/block3/unit_21/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_21')
         .relu(name='conv4_21_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_22/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_22/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_22/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_22/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_22/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_22/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_21_relu',
                   'pspnet_v1_101/block3/unit_22/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_22')
         .relu(name='conv4_22_relu')
         .conv(1, 1, 256, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_23/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_23/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block3/unit_23/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block3/unit_23/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 1024, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block3/unit_23/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block3/unit_23/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_22_relu',
                   'pspnet_v1_101/block3/unit_23/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv4_23')
         .relu(name='conv4_23_relu')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block4/unit_1/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block4/unit_1/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block4/unit_1/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block4/unit_1/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block4/unit_1/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block4/unit_1/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv4_23_relu')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block4/unit_1/bottleneck_v1/shortcut')
         .batch_normalization(name='pspnet_v1_101/block4/unit_1/bottleneck_v1/shortcut/BatchNorm'))

        (self.feed('pspnet_v1_101/block4/unit_1/bottleneck_v1/shortcut/BatchNorm',
                   'pspnet_v1_101/block4/unit_1/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv5_1')
         .relu(name='conv5_1_relu')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block4/unit_2/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block4/unit_2/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block4/unit_2/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block4/unit_2/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block4/unit_2/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block4/unit_2/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv5_1_relu',
                   'pspnet_v1_101/block4/unit_2/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv5_2')
         .relu(name='conv5_2_relu')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block4/unit_3/bottleneck_v1/conv1')
         .batch_normalization(relu=True, name='pspnet_v1_101/block4/unit_3/bottleneck_v1/conv1/BatchNorm')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False,
                      name='pspnet_v1_101/block4/unit_3/bottleneck_v1/conv2')
         .batch_normalization(relu=True, name='pspnet_v1_101/block4/unit_3/bottleneck_v1/conv2/BatchNorm')
         .conv(1, 1, 2048, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/block4/unit_3/bottleneck_v1/conv3')
         .batch_normalization(name='pspnet_v1_101/block4/unit_3/bottleneck_v1/conv3/BatchNorm'))

        (self.feed('conv5_2_relu',
                   'pspnet_v1_101/block4/unit_3/bottleneck_v1/conv3/BatchNorm')
         .add(name='conv5_3')
         .relu(name='conv5_3_relu'))

        conv5_3 = self.layers['conv5_3_relu']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('conv5_3_relu')
         .avg_pool(60, 60, 60, 60, padding='SAME', name='conv5_3_pool1')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/pyramid_pool_module/level1/pyramid_pool_v1/conv1')
         .batch_normalization(relu=True,
                              name='pspnet_v1_101/pyramid_pool_module/level1/pyramid_pool_v1/conv1/BatchNorm')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool1_interp')
         .interp(shape, name='conv5_3_pool1_interp')
         )

        (self.feed('conv5_3_relu')
         .avg_pool(30, 30, 30, 30, padding='SAME', name='conv5_3_pool2')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/pyramid_pool_module/level2/pyramid_pool_v1/conv1')
         .batch_normalization(relu=True,
                              name='pspnet_v1_101/pyramid_pool_module/level2/pyramid_pool_v1/conv1/BatchNorm')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool2_interp')
         .interp(shape, name='conv5_3_pool2_interp')
         )

        (self.feed('conv5_3_relu')
         .avg_pool(20, 20, 20, 20, padding='SAME', name='conv5_3_pool3')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/pyramid_pool_module/level3/pyramid_pool_v1/conv1')
         .batch_normalization(relu=True,
                              name='pspnet_v1_101/pyramid_pool_module/level3/pyramid_pool_v1/conv1/BatchNorm')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool3_interp')
         .interp(shape, name='conv5_3_pool3_interp')
         )

        (self.feed('conv5_3_relu')
         .avg_pool(10, 10, 10, 10, padding='SAME', name='conv5_3_pool6')
         .conv(1, 1, 512, 1, 1, padding='SAME', biased=False, relu=False,
               name='pspnet_v1_101/pyramid_pool_module/level4/pyramid_pool_v1/conv1')
         .batch_normalization(relu=True,
                              name='pspnet_v1_101/pyramid_pool_module/level4/pyramid_pool_v1/conv1/BatchNorm')
         # .interp(60, 60, 512, 1, 1, 0, 0, name='conv5_3_pool6_interp')
         .interp(shape, name='conv5_3_pool6_interp')
         )

        (self.feed('conv5_3_relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
         .concat(3, name='conv5_3_concat')
         .conv(3, 3, 512, 1, 1, padding='SAME', biased=False, relu=False, name='pspnet_v1_101/fc1')
         .batch_normalization(relu=True, name='pspnet_v1_101/fc1/BatchNorm')
         .conv(1, 1, 2, 1, 1, padding='SAME', relu=False, name='pspnet_v1_101/logits')
         # .interp(None, None, 21, 8, 1, 0, 0, name='conv6_interp')
         )
