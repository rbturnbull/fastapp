import enum
import types
from typing import get_type_hints, List
import torchvision.models as models
from torch import nn
from fastai.vision.learner import cnn_learner, unet_learner

from .apps import FastApp
from .params import Param


def torchvision_model_choices() -> List[str]:
    """
    Returns a list of function names in torchvision.models which can produce torch modules.

    For more information see: https://pytorch.org/vision/stable/models.html
    """
    model_choices = []
    for item in dir(models):
        obj = getattr(models, item)

        # Only accept functions
        if isinstance(obj, types.FunctionType):

            # Only accept if the return value is a pytorch module
            hints = get_type_hints(obj)
            return_value = hints.get("return", "")
            if nn.Module in return_value.mro():
                model_choices.append(item)
    return model_choices


TorchvisionModelEnum = enum.Enum(
    "TorchvisionModelName",
    {model_name: model_name for model_name in torchvision_model_choices()},
)


class VisionApp(FastApp):
    """
    A FastApp which uses a model from torchvision.
    """

    def default_model_name(self):
        return "resnet18"

    def model(
        self,
        model_name: TorchvisionModelEnum = Param(
            default="",
            help="The name of a model architecture in torchvision.models (https://pytorch.org/vision/stable/models.html). If not given, then it is given by `default_model_name`",
        ),
        pretrained: bool = Param(
            default=True, help="Whether or not to use the pretrained weights."
        ),
    ):
        if not model_name:
            model_name = self.default_model_name()

        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not recognized.")

        return getattr(models, model_name)(pretrained=pretrained)

    def build_learner_func(self):
        return cnn_learner


class UNetApp(VisionApp):
    """
    A FastApp which uses a base model from torchvision which is modified

    For more information see:
        Olaf Ronneberger, Philipp Fischer, Thomas Brox,
            U-Net: Convolutional Networks for Biomedical Image Segmentation,
            https://arxiv.org/abs/1505.04597
        https://fastai1.fast.ai/vision.learner.html#unet_learner
        https://github.com/fastai/fastbook/blob/master/15_arch_details.ipynb
    """

    def build_learner_func(self):
        return unet_learner
