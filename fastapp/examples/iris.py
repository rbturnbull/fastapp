import numpy as np
from torch import nn
from fastai.tabular.data import TabularDataLoaders
from sklearn.datasets import load_iris

import fastapp as fa

class IrisApp(fa.FastApp):
    def __init__(self):
        self.data = load_iris(as_frame=True)

    def dataloaders(
        self,
        batch_size:int = fa.Param(32,  tune_min=8, tune_max=128, log=True),
    ):
        df = self.data['frame']
        df['target_name'] = np.take(self.data['target_names'], df["target"])
        
        return TabularDataLoaders.from_df(
            self.data['frame'],
            cont_names=self.data['feature_names'],
            y_names='target_name',
            bs=batch_size,
        )

    def model(
        self,
        hidden_size:int = fa.Param(128, tune_min=8, tune_max=1028, log=True, help="The number of hidden layers."),
    ):
        return nn.Sequential(
            nn.Linear(in_features=len(self.data['feature_names']), out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=len(self.data['target_names'])),
        )