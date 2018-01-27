import math
from collections import namedtuple

from .errors import KaffeError

TensorShape = namedtuple('TensorShape', ['batch_size', 'channels', 'height', 'width'])


def get_filter_output_shape(i_h, i_w, params, round_func, do_dilation=False):
    if do_dilation:
        assert params.stride_h == 1 and params.stride_w == 1
        kernel_h = params.kernel_h + (params.kernel_h - 1) * (params.dilation_h - 1)
        kernel_w = params.kernel_w + (params.kernel_w - 1) * (params.dilation_w - 1)

        o_h = (i_h + 2 * params.pad_h - kernel_h) / float(params.stride_h) + 1
        o_w = (i_w + 2 * params.pad_w - kernel_w) / float(params.stride_w) + 1
    else:
        o_h = (i_h + 2 * params.pad_h - params.kernel_h) / float(params.stride_h) + 1
        o_w = (i_w + 2 * params.pad_w - params.kernel_w) / float(params.stride_w) + 1
    return int(round_func(o_h)), int(round_func(o_w))


def get_strided_kernel_output_shape(node, round_func):
    assert node.layer is not None
    input_shape = node.get_only_parent().output_shape
    node.layer.set_input_shape(input_shape)

    do_dilation = False
    if hasattr(node.layer.kernel_parameters, 'dilation_h'):
        if node.layer.kernel_parameters.dilation_h > 1:
            do_dilation = True

    o_h, o_w = get_filter_output_shape(input_shape.height, input_shape.width,
                                       node.layer.kernel_parameters, round_func,
                                       do_dilation=do_dilation)
    params = node.layer.parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels
    return TensorShape(input_shape.batch_size, c, o_h, o_w)


def get_interp_output_shape_impl(i_h, i_w, params, round_func):
    shrink_factor = params.shrink_factor
    zoom_factor = params.zoom_factor
    height = params.height
    width = params.width
    pad_beg = params.pad_beg
    pad_end = params.pad_end

    height_in_eff_ = int(i_h + pad_beg + pad_end)
    width_in_eff_ = int(i_w + pad_beg + pad_end)

    assert height_in_eff_ > 0, 'height should be positive'
    assert width_in_eff_ > 0, 'width should be positive'

    if shrink_factor is not None and zoom_factor is None:
        assert shrink_factor >= 1, 'Shrink factor must be positive'
        o_h = (height_in_eff_ - 1) / shrink_factor + 1
        o_w = (width_in_eff_ - 1) / shrink_factor + 1
    elif shrink_factor is None and zoom_factor is not None:
        assert zoom_factor >= 1, 'Shrink factor must be positive'
        o_h = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1)
        o_w = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1)
    elif height is not None and width is not None:
        o_h = height
        o_w = width
    elif shrink_factor is not None and zoom_factor is not None:
        assert shrink_factor >= 1, 'Shrink factor must be positive'
        assert zoom_factor >= 1, 'Zoom factor must be positive'

        o_h = (height_in_eff_ - 1) / shrink_factor + 1
        o_w = (width_in_eff_ - 1) / shrink_factor + 1
        o_h = o_h + (o_h - 1) * (zoom_factor - 1)
        o_w = o_w + (o_w - 1) * (zoom_factor - 1)
    else:
        o_h = 0
        o_w = 0
        KaffeError('InterpLayer error. Should not come here')

    assert o_h > 0, 'height should be positive'
    assert o_w > 0, 'width should be positive'

    return int(round_func(o_h)), int(round_func(o_w))


def get_interp_output_shape(node, round_func):
    assert node.layer is not None
    input_shape = node.get_only_parent().output_shape
    node.layer.set_input_shape(input_shape)
    o_h, o_w = get_interp_output_shape_impl(input_shape.height, input_shape.width,
                                            node.layer.interp_parameters, round_func)

    return TensorShape(input_shape.batch_size, input_shape.channels, o_h, o_w)


def shape_not_implemented(node):
    raise NotImplementedError


def shape_identity(node):
    assert len(node.parents) > 0
    return node.parents[0].output_shape


def shape_scalar(node):
    return TensorShape(1, 1, 1, 1)


def shape_data(node):
    if node.output_shape:
        # Old-style input specification
        return node.output_shape
    try:
        # New-style input specification
        return map(int, node.parameters.shape[0].dim)
    except:
        # We most likely have a data layer on our hands. The problem is,
        # Caffe infers the dimensions of the data from the source (eg: LMDB).
        # We want to avoid reading datasets here. Fail for now.
        # This can be temporarily fixed by transforming the data layer to
        # Caffe's "input" layer (as is usually used in the "deploy" version).
        # TODO: Find a better solution for this.
        raise KaffeError('Cannot determine dimensions of data layer.\n'
                         'See comments in function shape_data for more info.')


def shape_mem_data(node):
    params = node.parameters
    return TensorShape(params.batch_size, params.channels, params.height, params.width)


def shape_concat(node):
    axis = node.layer.parameters.axis
    output_shape = None
    for parent in node.parents:
        if output_shape is None:
            output_shape = list(parent.output_shape)
        else:
            output_shape[axis] += parent.output_shape[axis]
    return tuple(output_shape)


def shape_convolution(node):
    return get_strided_kernel_output_shape(node, math.floor)


def shape_pool(node):
    return get_strided_kernel_output_shape(node, math.ceil)


def shape_inner_product(node):
    input_shape = node.get_only_parent().output_shape
    return TensorShape(input_shape.batch_size, node.layer.parameters.num_output, 1, 1)


def shape_interp(node):
    return get_interp_output_shape(node, math.floor)
