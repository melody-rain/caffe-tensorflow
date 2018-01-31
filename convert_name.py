import os
import re
import json

data_src = '/home/melody/develop/caffe-tensorflow/caffe_name.txt'

param_map = {'variance': 'moving_variance',
             'scale': 'gamma',
             'offset': 'beta',
             'mean': 'moving_mean',
             'weights': 'weights'}

psp_map = {
    '1': '1',
    '2': '2',
    '3': '3',
    '6': '4'}

caffe_tf_name = {}

with open(data_src) as fd:
    lines = fd.readlines()
    lines = [[a_line.strip().split()[0], a_line.strip().split()[1]] for a_line in lines]

tf_prefix = 'pspnet_v1_101'

for i, a_line in enumerate(lines):
    a_name = a_line[0]
    param = a_line[1]
    assert a_name.startswith('conv'), a_name
    is_psp = False
    if True:
        add_conv_id = False
        match = re.search('\d+_\d+_\d+x\d+', a_name)
        if not match:
            if 'pool' in a_name:
                pattern = '\d+_\d+_pool\d+'
                is_psp = True
            elif a_name.startswith('conv5_4'):
                pattern = 'conv5_4'
                pattern = '5_4'
            # elif a_name.startswith('conv6'):
            #     pattern = 'conv6'
            # elif a_name.startswith('conv_aux'):
            #     pattern = 'conv_aux'
            else:
                print 'invalid', a_name
                continue
            match = re.search(pattern, a_name)
        else:
            if 'reduce' in a_name:
                conv_id = 1
            elif 'increase' in a_name:
                conv_id = 3
            else:
                conv_id = 2

            add_conv_id = True

        postfix = a_name[match.span()[1]:]

        info = match.group()
        print '<<<', info, a_name, match.span()

        # info = re.findall('\d', info)
        info = re.split('[\s,.,x_]', info)
        # info = re.split('[\D]', info)
        print '>>>',info
        block_id = int(info[0])
        unit_id = int(info[1])

        tf_block_id = 'block{}'.format(block_id - 1)
        if add_conv_id:
            op_name = 'conv{}'.format(conv_id)
        else:
            op_name = 'conv{}'.format(unit_id)

        if 'proj' in postfix:
            op_name = 'shortcut'

        if is_psp:
            tf_block_id = 'pyramid_pool_module/level{}/pyramid_pool_v1'.format(psp_map[info[2][-1]])
            op_name = 'conv1'
        elif a_name.startswith('conv5_4'):
            tf_block_id = 'fc1'
            op_name = ''
        else:
            if block_id == 1:
                tf_block_id = 'root'
                op_name = 'conv{}'.format(unit_id)
            else:
                tf_block_id = '{}/unit_{}/bottleneck_v1'.format(tf_block_id, unit_id)

        if postfix.endswith('bn'):
            if op_name == '':
                op_name = 'BatchNorm'
            else:
                op_name = '{}/BatchNorm'.format(op_name)

        if op_name == '':
            # tf_name = '{}/{}/{}'.format(tf_prefix, tf_block_id, param_map[param])
            tf_name = '{}/{}'.format(tf_prefix, tf_block_id)
        else:
            # tf_name = '{}/{}/{}/{}'.format(tf_prefix, tf_block_id, op_name, param_map[param])
            tf_name = '{}/{}/{}'.format(tf_prefix, tf_block_id, op_name)
        print a_name, ' ---> ', tf_name
        caffe_tf_name[a_name] = tf_name
    else:
        print '->', a_name

caffe_tf_name['conv6'] = 'pspnet_v1_101/logits'
caffe_tf_name['conv_aux'] = 'pspnet_v1_101/aux_logits'

with open('pspnet_dict.json', 'w') as fd:
    json.dump(caffe_tf_name, fd, sort_keys=True, indent=4)