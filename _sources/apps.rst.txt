=======================
Apps
=======================

Inherit a class from FastApp to make an app. The parent class includes several methods for training and hyper-parameter tuning. 
The minimum requirement is that you fill out the dataloaders method and the model method.

The dataloaders method requires that you return a fastai dataloaders object. This is a collection of dataloader objects. 
Typically it contains one dataloader for training and another for testing. For more information see https://docs.fast.ai/data.core.html#DataLoaders
You can add parameter values with typing hints in the function signature and these will be automatically added to the train and show_batch methods.


The model method requires that you return a pytorch module. Parameters in the function signature will be added to the train method.

Command-Line Interface
=======================

To 

Add more command-line commands by overriding cli method.
