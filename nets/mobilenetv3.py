from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large



class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
class MobileV3Unet(nn.Module):
    def __init__(self,  pretrain_backbone: bool = False):
        super(MobileV3Unet, self).__init__()
        backbone = mobilenet_v3_large(pretrained=pretrain_backbone)

        # if pretrain_backbone:
        #     # 载入mobilenetv3 large backbone预训练权重
        #     # https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
        #     backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

        backbone = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        feat1 = backbone_out["stage0"]
        feat2 = backbone_out["stage1"]
        feat3 = backbone_out["stage2"]
        feat4 = backbone_out["stage3"]
        feat5 = backbone_out["stage4"]

        return [feat1, feat2, feat3, feat4, feat5]
def mobilenet(pretrained=False, **kwargs):
    model = MobileV3Unet(pretrain_backbone=False)
    if pretrained:
        None
        pass
    return model
# net = mobilenet
# from torchinfo import summary
# summary(net(), input_size=(1, 3, 224, 224))


