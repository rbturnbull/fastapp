#!/usr/bin/env python3
import numpy as np
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.all import tabular_learner
from sklearn.datasets import load_iris
import fastapp as fa


class IrisApp(fa.FastApp):
    """
    A classification app to predict the type of iris from sepal and petal lengths and widths.

    A classic dataset publised in:
        Fisher, R.A. “The use of multiple measurements in taxonomic problems” Annual Eugenics, 7, Part II, 179-188 (1936).
    For more information about the dataset, see:
        https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
    """

    def dataloaders(
        self,
        batch_size: int = fa.Param(32, tune_min=8, tune_max=128, log=True),
    ):
        df = load_iris(as_frame=True)
        df["target_name"] = np.take(df["target_names"], df["target"])

        return TabularDataLoaders.from_df(
            df["frame"],
            cont_names=df["feature_names"],
            y_names="target_name",
            bs=batch_size,
        )

    def model(self):
        return None

    def build_learner_func(self):
        return tabular_learner


# class IrisWandbApp(fastapp.logging.WandbMixin, IrisApp):
#     """
#     A version of the iris app which also includes logging to 'Weights & Biases' (https://wandb.ai/).
#     """

if __name__ == "__main__":
    IrisApp().main()
