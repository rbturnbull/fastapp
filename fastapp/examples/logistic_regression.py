from pathlib import Path
import pandas as pd
from torch import nn
from fastai.data.block import DataBlock, TransformBlock, CategoryBlock
from fastai.data.transforms import ColReader, RandomSplitter
import fastapp as fa
from fastai.metrics import accuracy


class LogisticRegressionApp(fa.FastApp):
    def dataloaders(
        self,
        csv:Path = fa.Param(help="The path to a CSV file with the data."),
        x:str = fa.Param(default="x", help="The column name of the independent variable."),
        y:str = fa.Param(default="y", help="The column name of the dependent variable."),
        validation_proportion:float = fa.Param(default=0.2, help="The column name of the dependent variable."),
        batch_size:int = fa.Param(default=32, tune=True, tune_min=8, tune_max=128, log=True),
    ):
        def unsqueeze(tensor):
            return tensor.unsqueeze(-1).float()

        datablock = DataBlock(
            blocks=[TransformBlock, CategoryBlock],
            get_x=ColReader(x),
            get_y=ColReader(y),
            splitter=RandomSplitter(validation_proportion),
            batch_tfms=unsqueeze,
        )
        df = pd.read_csv(csv)
        
        return datablock.dataloaders(df,bs=batch_size)

    def model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features=1, out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def loss_func(self):
        return nn.BCELoss()

    def metrics(self):
        return [accuracy]