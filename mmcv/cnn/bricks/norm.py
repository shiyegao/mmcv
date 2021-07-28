from copy import deepcopy
import inspect
import torch
import torch.nn as nn

from mmcv.utils import is_tuple_of
from mmcv.utils.parrots_wrapper import SyncBatchNorm, _BatchNorm, _InstanceNorm
from .registry import NORM_LAYERS


class ResetMeanVarBatchNorm2d(nn.BatchNorm2d):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self.reset_running_stats()


class OnlineMeanVarBatchNorm2d(nn.Module):
    __constants__ = ['num_features', 'momentum']
    def __init__(self, num_features, eps, momentum=0.1):
        # super().__init__(num_features)
        super().__init__()
        assert eps==1e-5
        self.num_features, self.momentum = num_features, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - 1e-5)


    def extra_repr(self):
        return '{num_features}, eps=1e-5, momentum={momentum}, affine=True'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, #strict,
            missing_keys, unexpected_keys, error_msgs)


    def forward(self, x):
        current_mean = x.mean([0, 2, 3])
        current_var = x.var([0, 2, 3], unbiased=False)

        scale = self.weight * ((1 - self.momentum
            ) * self.running_var + current_var * self.momentum + 1e-5).rsqrt()
        bias = self.bias - ((1 - self.momentum
        ) * self.running_mean + current_mean * self.momentum) * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        return x * scale + bias
    

class CurrentMeanVarBatchNorm2d(nn.Module):
    __constants__ = ['num_features']
    def __init__(self, num_features, eps):
        # super().__init__(num_features)
        super().__init__()
        assert eps==1e-5
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - 1e-5)
        # self.running_mean = torch.zeros(num_features)
        # self.running_var = torch.ones(num_features) - 1e-5

    def extra_repr(self):
        return '{num_features}, eps=1e-5, affine=True'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        model_device = self.weight.device
        self.weight = nn.Parameter(torch.ones(self.num_features).to(model_device))
        self.bias = nn.Parameter(torch.zeros(self.num_features).to(model_device))

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, #strict,
            missing_keys, unexpected_keys, error_msgs)
        
        self.register_buffer("ckpt_weight", self.weight)
        self.register_buffer("ckpt_bias", self.bias)


    def forward(self, x):
        current_mean = x.mean([0, 2, 3])
        current_var = x.var([0, 2, 3], unbiased=False)
        if self.weight.dim() == 1: # batch
            scale = self.weight * (current_var + 1e-5).rsqrt()
            bias = self.bias - current_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
        else: # sample
            scale = self.weight * ((current_var + 1e-5).rsqrt()).reshape(1, -1)
            bias = self.bias - current_mean.reshape(1, -1) * scale
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class FreezedMeanVarBatchNorm2d(nn.Module):
    __constants__ = ['num_features']
    def __init__(self, num_features, eps):
        # super().__init__(num_features)
        super().__init__()
        assert eps==1e-5
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - 1e-5)
        # self.running_mean = torch.zeros(num_features)
        # self.running_var = torch.ones(num_features) - 1e-5

    def extra_repr(self):
        return '{num_features}, eps=1e-5, affine=True'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        model_device = self.weight.device
        self.weight = nn.Parameter(torch.ones(self.num_features).to(model_device))
        self.bias = nn.Parameter(torch.zeros(self.num_features).to(model_device))

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, #strict,
            missing_keys, unexpected_keys, error_msgs)

        self.register_buffer("ckpt_weight", self.weight)
        self.register_buffer("ckpt_bias", self.bias)

    def forward(self, x):
        if self.weight.dim() == 1: # batch
            scale = self.weight * (self.running_var + 1e-5).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else: # sample
            scale = self.weight * ((self.running_var + 1e-5).rsqrt()).reshape(1, -1)
            bias = self.bias - self.running_mean.reshape(1, -1) * scale
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class TargetCurrentMeanVarBatchNorm2d(nn.Module):
    __constants__ = ['num_features']
    def __init__(self, num_features, eps):
        super().__init__()
        assert eps==1e-5
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - 1e-5)
        self.register_buffer("current_mean", torch.zeros(num_features))
        self.register_buffer("current_var", torch.ones(num_features) - 1e-5)
        self.register_buffer("target_mean", torch.zeros(num_features))
        self.register_buffer("target_var", torch.ones(num_features) - 1e-5)

    def extra_repr(self):
        return '{num_features}, eps=1e-5, affine=True'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        model_device = self.weight.device
        self.weight = nn.Parameter(torch.ones(self.num_features).to(model_device))
        self.bias = nn.Parameter(torch.zeros(self.num_features).to(model_device))

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, #strict,
            missing_keys, unexpected_keys, error_msgs)
        
        self.target_mean = deepcopy(self.running_mean)
        self.target_var = deepcopy(self.running_var)


    def forward(self, x):
        self.current_mean = x.mean([0, 2, 3])
        self.current_var = x.var([0, 2, 3], unbiased=False)
        if self.weight.dim() == 1: # batch
            scale = self.weight * (self.current_var + 1e-5).rsqrt()
            bias = self.bias - self.current_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
        else: # sample
            scale = self.weight * ((self.current_var + 1e-5).rsqrt()).reshape(1, -1)
            bias = self.bias - self.current_mean.reshape(1, -1) * scale
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class TargetFreezedMeanVarBatchNorm2d(nn.Module):
    __constants__ = ['num_features']
    def __init__(self, num_features, eps):
        super().__init__()
        assert eps==1e-5
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - 1e-5)
        self.register_buffer("current_mean", torch.zeros(num_features))
        self.register_buffer("current_var", torch.ones(num_features) - 1e-5)
        self.register_buffer("target_mean", torch.zeros(num_features))
        self.register_buffer("target_var", torch.ones(num_features) - 1e-5)

    def extra_repr(self):
        return '{num_features}, eps=1e-5, affine=True'.format(**self.__dict__)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        model_device = self.weight.device
        self.weight = nn.Parameter(torch.ones(self.num_features).to(model_device))
        self.bias = nn.Parameter(torch.zeros(self.num_features).to(model_device))

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, #strict,
            missing_keys, unexpected_keys, error_msgs)

        self.target_mean = deepcopy(self.running_mean)
        self.target_var = deepcopy(self.running_var)

    def forward(self, x):
        self.current_mean = x.mean([0, 2, 3])
        self.current_var = x.var([0, 2, 3], unbiased=False)

        if self.weight.dim() == 1: # batch
            scale = self.weight * (self.running_var + 1e-5).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else: # sample
            scale = self.weight * ((self.running_var + self.eps).rsqrt()).reshape(1, -1)
            bias = self.bias - self.running_mean.reshape(1, -1) * scale
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


NORM_LAYERS.register_module('BN', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN1d', module=nn.BatchNorm1d)
NORM_LAYERS.register_module('BN2d', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN3d', module=nn.BatchNorm3d)
NORM_LAYERS.register_module('SyncBN', module=SyncBatchNorm)
NORM_LAYERS.register_module('GN', module=nn.GroupNorm)
NORM_LAYERS.register_module('LN', module=nn.LayerNorm)
NORM_LAYERS.register_module('IN', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN1d', module=nn.InstanceNorm1d)
NORM_LAYERS.register_module('IN2d', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN3d', module=nn.InstanceNorm3d)

NORM_LAYERS.register_module('cmvBN', module=CurrentMeanVarBatchNorm2d)
NORM_LAYERS.register_module('fmvBN', module=FreezedMeanVarBatchNorm2d)
NORM_LAYERS.register_module('omvBN', module=OnlineMeanVarBatchNorm2d)
NORM_LAYERS.register_module('rmvBN', module=ResetMeanVarBatchNorm2d)
NORM_LAYERS.register_module('cmvBN2d', module=CurrentMeanVarBatchNorm2d)
NORM_LAYERS.register_module('fmvBN2d', module=FreezedMeanVarBatchNorm2d)
NORM_LAYERS.register_module('omvBN2d', module=OnlineMeanVarBatchNorm2d)
NORM_LAYERS.register_module('rmvBN2d', module=ResetMeanVarBatchNorm2d)

def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm(layer, exclude=None):
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)
