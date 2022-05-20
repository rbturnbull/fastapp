from pathlib import Path
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import ColReader, RandomSplitter, DisplayedTransform
from fastai.metrics import accuracy
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize, ResizeMethod
import pandas as pd
import fastapp as fa

from fastapp.vision import VisionApp

class PathColReader(DisplayedTransform):
    def __init__(self, column_name: str, base_dir: Path):
        self.column_name = column_name
        self.base_dir = base_dir

    def __call__(self, row, **kwargs):
        path = Path(row[self.column_name])
        if not path.is_absolute():
            path = self.base_dir / path
        return path


class ImageClassifier(VisionApp):
    """
    A FastApp for classifying images.

    For training, it expects a CSV with image paths and categories.
    """

    def dataloaders(
        self,
        csv: Path = fa.Param(default=None, help="A CSV with image paths and categories."),
        image_column: str = fa.Param(default="image", help="The name of the column with the image paths."),
        category_column: str = fa.Param(
            default="category", help="The name of the column with the category of the image."
        ),
        base_dir: Path = fa.Param(default="./", help="The base directory for images with relative paths."),
        validation_proportion: float = fa.Param(
            default=0.2, help="The proportion of the dataset to keep for validation"
        ),
        batch_size: int = fa.Param(default=16, help="The number of items to use in each batch."),
        width: int = fa.Param(default=224, help="The width to resize all the images to."),
        height: int = fa.Param(default=224, help="The height to resize all the images to."),
        resize_method:str = fa.Param(default="squish", help="The method to resize images."),
    ):
        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=PathColReader(column_name=image_column, base_dir=base_dir),
            get_y=ColReader(category_column),
            splitter=RandomSplitter(validation_proportion),
            item_tfms=Resize( (width,height), method=resize_method ),
        )
        df = pd.read_csv(csv)

        return datablock.dataloaders(df, bs=batch_size)

    def metrics(self):
        return [accuracy]

    def monitor(self):
        return "accuracy"


if __name__ == "__main__":
    ImageClassifier.main()
