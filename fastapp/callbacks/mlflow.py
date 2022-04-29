from fastai.callback.core import Callback


class FastAppMlflowCallback(Callback):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
