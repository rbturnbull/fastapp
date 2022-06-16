import torch
from fastai.data.block import TransformBlock


def bool_to_tensor(value: bool):
    return torch.FloatTensor(value)


def unsqueeze(inputs):
    """This is needed to transform the input with an extra dimension added to the end of the tensor."""
    return inputs.unsqueeze(dim=-1).float()


def BoolBlock():
    return TransformBlock(
        item_tfms=[bool_to_tensor],
        batch_tfms=unsqueeze,
    )
