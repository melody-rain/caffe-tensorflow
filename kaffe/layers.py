import re
import numbers
from collections import namedtuple

from .shapes import *

LAYER_DESCRIPTORS = {

    # Caffe Types
    'AbsVal': shape_identity,
    'Accuracy': shape_scalar,
    'ArgMax': shape_not_implemented,
    'BatchNorm': shape_identity,
    'BNLL': shape_not_implemented,
    'Concat': shape_concat,
    'ContrastiveLoss': shape_scalar,
    'Convolution': shape_convolution,
    'Deconvolution': shape_not_implemented,
    'Data': shape_data,
    'Dropout': shape_identity,
    'DummyData': shape_data,
    'EuclideanLoss': shape_scalar,
    'Eltwise': shape_identity,
    'Exp': shape_identity,
    'Flatten': shape_not_implemented,
    'HDF5Data': shape_data,
    'HDF5Output': shape_identity,
    'HingeLoss': shape_scalar,
    'Im2col': shape_not_implemented,
    'ImageData': shape_data,
    'InfogainLoss': shape_scalar,
    'InnerProduct': shape_inner_product,
    'Input': shape_data,
    'LRN': shape_identity,
    'MemoryData': shape_mem_data,
    'MultinomialLogisticLoss': shape_scalar,
    'MVN': shape_not_implemented,
    'Pooling': shape_pool,
    'Power': shape_identity,
    'ReLU': shape_identity,
    'Scale': shape_identity,
    'Sigmoid': shape_identity,
    'SigmoidCrossEntropyLoss': shape_scalar,
    'Silence': shape_not_implemented,
    'Softmax': shape_identity,
    'SoftmaxWithLoss': shape_scalar,
    'Split': shape_not_implemented,
    'Slice': shape_not_implemented,
    'TanH': shape_identity,
    'WindowData': shape_not_implemented,
    'Threshold': shape_identity,
    'BN': shape_identity,
    'Interp': shape_interp,
}

LAYER_TYPES = LAYER_DESCRIPTORS.keys()

LayerType = type('LayerType', (), {t: t for t in LAYER_TYPES})


class NodeKind(LayerType):
    @staticmethod
    def map_raw_kind(kind):
        if kind in LAYER_TYPES:
            return kind
        return None

    @staticmethod
    def compute_output_shape(node):
        try:
            val = LAYER_DESCRIPTORS[node.kind](node)
            return val
        except NotImplementedError:
            raise KaffeError(__file__, 'Output shape computation not implemented for type: %s' % node.kind)


class NodeDispatchError(KaffeError):
    pass


class NodeDispatch(object):
    @staticmethod
    def get_handler_name(node_kind):
        if node_kind == 'BN':
            return node_kind
        elif len(node_kind) <= 4:
            # A catch-all for things like ReLU and tanh
            return node_kind.lower()
        # Convert from CamelCase to under_scored
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', node_kind)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def get_handler(self, node_kind, prefix):
        name = self.get_handler_name(node_kind)
        name = '_'.join((prefix, name))
        try:
            return getattr(self, name)
        except AttributeError:
            raise NodeDispatchError('No handler found for node kind: %s (expected: %s)' %
                                    (node_kind, name))


class LayerAdapter(object):
    def __init__(self, layer, kind):
        self.layer = layer
        self.kind = kind
        self._input_shape = None

    @property
    def parameters(self):
        name = NodeDispatch.get_handler_name(self.kind)
        name = '_'.join((name, 'param'))
        try:
            return getattr(self.layer, name)
        except AttributeError:
            raise NodeDispatchError('Caffe parameters not found for layer kind: %s' % (self.kind))

    @staticmethod
    def get_kernel_value(scalar, repeated, idx, default=None):
        if scalar:
            return scalar
        if repeated:
            if isinstance(repeated, numbers.Number):
                return repeated
            if len(repeated) == 1:
                # Same value applies to all spatial dimensions
                return int(repeated[0])
            assert idx < len(repeated)
            # Extract the value for the given spatial dimension
            return repeated[idx]
        if default is None:
            raise ValueError('Unable to determine kernel parameter!')
        return default

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    @property
    def kernel_parameters(self):
        assert self.kind in (NodeKind.Convolution, NodeKind.Pooling)
        params = self.parameters
        global_pool = hasattr(params, 'global_pooling')

        if params.kernel_size:
            k_h = self.get_kernel_value(params.kernel_h, params.kernel_size, 0)
            k_w = self.get_kernel_value(params.kernel_w, params.kernel_size, 1)
        elif self._input_shape:
            k_h, k_w = [self._input_shape.height, self._input_shape.width]
        else:  # errors out in get_kernel_value function
            k_h = self.get_kernel_value(params.kernel_h, params.kernel_size, 0)
            k_w = self.get_kernel_value(params.kernel_w, params.kernel_size, 1)

        s_h = self.get_kernel_value(params.stride_h, params.stride, 0, default=1)
        s_w = self.get_kernel_value(params.stride_w, params.stride, 1, default=1)
        p_h = self.get_kernel_value(params.pad_h, params.pad, 0, default=0)
        p_w = self.get_kernel_value(params.pad_h, params.pad, 1, default=0)
        return KernelParameters(k_h, k_w, s_h, s_w, p_h, p_w)

    @property
    def interp_parameters(self):
        assert self.kind == NodeKind.Interp, 'Must be for interp layer'
        params = self.parameters

        shrink_factor = params.shrink_factor if params.shrink_factor else None
        print 'dddd', params.shrink_factor
        zoom_factor = params.zoom_factor if params.zoom_factor else None
        height = params.height if params.height else None
        width = params.width if params.width else None
        pad_beg = params.pad_beg
        pad_end = params.pad_end
        assert pad_beg <= 0, 'Only supports non-pos padding (cropping) for now'
        assert pad_end <= 0, 'Only supports non-pos padding (cropping) for now'

        print 'interp_parameters', params, self.layer
        return InterpParameters(height, width, zoom_factor, shrink_factor, pad_beg, pad_end)


KernelParameters = namedtuple('KernelParameters', ['kernel_h', 'kernel_w', 'stride_h', 'stride_w',
                                                   'pad_h', 'pad_w'])

InterpParameters = namedtuple('InterpParameters', ['height', 'width', 'zoom_factor',
                                                   'shrink_factor', 'pad_beg', 'pad_end'])