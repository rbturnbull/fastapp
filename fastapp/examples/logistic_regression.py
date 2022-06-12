#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from torch import nn
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.transforms import ColReader, RandomSplitter
import fastapp as fa
from fastapp.blocks import BoolBlock
from fastapp.metrics import logit_accuracy, logit_f1


class LogisticRegressionApp(fa.FastApp):
    """
    Creates a basic app to do logistic regression.
    """

    def dataloaders(
        self,
        csv: Path = fa.Param(help="The path to a CSV file with the data."),
        x: str = fa.Param(default="x", help="The column name of the independent variable."),
        y: str = fa.Param(default="y", help="The column name of the dependent variable."),
        validation_proportion: float = fa.Param(
            default=0.2, help="The proportion of the dataset to use for validation."
        ),
        batch_size: int = fa.Param(
            default=32,
            tune=True,
            tune_min=8,
            tune_max=128,
            log=True,
            help="The number of items to use in each batch.",
        ),
    ):

        datablock = DataBlock(
            blocks=[TransformBlock, BoolBlock],
            get_x=ColReader(x),
            get_y=ColReader(y),
            splitter=RandomSplitter(validation_proportion),
        )
        df = pd.read_csv(csv)

        return datablock.dataloaders(df, bs=batch_size)

    def model(self) -> nn.Module:
        """Builds a simple logistic regression model."""
        return nn.Linear(in_features=1, out_features=1, bias=True)

    def loss_func(self):
        return nn.BCEWithLogitsLoss()

    def metrics(self):
        return [logit_accuracy, logit_f1]

    def monitor(self):
        return "logit_f1"


if __name__ == "__main__":
    LogisticRegressionApp.main()
