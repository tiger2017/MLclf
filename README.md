![mini-ImageNet Logo](https://github.com/tiger2017/MLclf/blob/master/mini-imagenet.png)
![mini-ImageNet Logo](https://github.com/tiger2017/MLclf/blob/master/tiny-imagenet.png)
## The project Machine Learning CLassiFication (MLclf) 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/mlclf)](https://pypi.org/project/mlclf/)
[![Downloads](https://static.pepy.tech/personalized-badge/mlclf?period=total&units=international_system&left_color=blue&right_color=green&left_text=Downloads)](https://pepy.tech/project/mlclf)

#### The purpose of this project is:
> 1. transform the mini-imagenet dataset which is initially created for the few-shot learning to the format that fit the classical classification task. You can also use this package to download and obtain the raw data of the mini-imagenet dataset (for few-shot learning tasks).

> 2. transform the tiny-imagenet dataset to the format that fit the classical classification task, which can be more easily used (being able to directly input to the Pytorch dataloader) compared to the original raw format.

> 3. tranform other popular datasets to the format that fit the classical classification task or the few-shot learning / zero-shot learning / transfer learning tasks (Feel free to leave your message if your have any ideas for the selection of potential datasets).

#### The original dataset of mini-imagenet includes totally 100 classes, but due to its intention to meta-learning or few-shot learning, the train/validation/test dataset contains different classes. They have respectively 64/16/20 classes.
#### The original dataset of tiny-imagenet includes totally 200 classes, the train/validation/test dataset contains all classes. They have respectively 100000/10000/10000 images. For example, the training dataset has 500 images for each class.

#### In order to make the mini/tiny-imagenet dataset fit the format requirement for the classical classification task. MLclf made a proper transformation (recombination and splitting) of the original mini/tiny-imagenet dataset.
#### The transformed dataset of mini-imagenet is divided into train, validation and test dataset, each dataset of which includes 100 classes. Each image has the size 84x84 pixels with 3 channels.
#### The transformed dataset of tiny-imagenet is divided into train, validation and test dataset, each dataset of which includes 200 classes. Each image has the size 64x64 pixels with 3 channels.

#### Notice: The provider of tiny-imagenet dataset does not public the labels of testing dataset, so there is no labels for the original raw testing dataset.

The MLclf package can be found at: https://github.com/tiger2017/MLclf 
                            or at: https://pypi.org/project/MLclf/

Welcome to create an issue to the repository of MLclf on GitHub, and I will add more datasets loading functions based on the issues.


The mini-imagenet source data can be also accessed from: https://deepai.org/dataset/imagenet (there is no need to manually download it if you use MLclf).


### Summary

* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)

### Requirements

- Python 3.x
- numpy
- torchvision



### Installation
How to install MLclf package:
```angular2html
pip install MLclf
```

### Usage
How to use this package for **mini-imagenet**:
```python
from MLclf import MLclf
import torch
import torchvision.transforms as transforms

# Download the original mini-imagenet data:
MLclf.miniimagenet_download(Download=True) # only need to run this line before you download the mini-imagenet dataset for the first time.

# Transform the original data into the format that fits the task for classification:
# Note: If you want to keep the data format as the same as that for the meta-learning or few-shot learning (original format), just set ratio_train=0.64, ratio_val=0.16, shuffle=False.

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# The argument transform is a optional keyword. You can also set transform = None or simply not set transform, if you do not want the data being standardized and only want a normalization b/t [0,1].
# The line below transformed the mini-imagenet data into the format for the traditional classification task, e.g. 60% training, 20% validation and 20% testing, with 100 classes in each of training/validation/testing set.
train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, transform=transform, save_clf_data=True)

# The dataset can be easily converted to dataloader via torch: 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0)


# You can check the corresponding relations between labels and label_marks of the image data:
# (Note: The relations below can be obtained after MLclf.miniimagenet_clf_dataset is called, otherwise they will be returned as None instead.)

labels_to_marks = MLclf.labels_to_marks['mini-imagenet']
marks_to_labels = MLclf.marks_to_labels['mini-imagenet']
```
####

You can also obtain the raw data of **mini-imagenet** from the downloaded pkl files:
```python
from MLclf import MLclf

# The raw data of mini-imagenet can be also obtained via the function below:

data_raw_train, data_raw_val, data_raw_test = MLclf.miniimagenet_data_raw()
```

How to use this package for **tiny-imagenet** for the traditional classification task (similarly as mini-imagenet):
```python
from MLclf import MLclf
import torch
import torchvision.transforms as transforms

MLclf.tinyimagenet_download(Download=True) # only need to run this line before you download the tiny-imagenet dataset for the first time.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset, validation_dataset, test_dataset = MLclf.tinyimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2,
                                                                                     seed_value=None, shuffle=True,
                                                                                     transform=transform,
                                                                                     save_clf_data=True,
                                                                                     few_shot=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, num_workers=0)

# You can check the corresponding relations between labels and label_marks of the image data:
# (Note: The relations below can be obtained after MLclf.miniimagenet_clf_dataset is called, otherwise they will be returned as None instead.)

labels_to_marks = MLclf.labels_to_marks['tiny-imagenet']
marks_to_labels = MLclf.marks_to_labels['tiny-imagenet']


data_raw_train, data_raw_val, data_raw_test = MLclf.tinyimagenet_data_raw()
```

If you want to use **tiny-imagenet** for the few-shot learning task, just change few_shot=True, for example:
```python
train_dataset, validation_dataset, test_dataset = MLclf.tinyimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2,
                                                                                     seed_value=None, shuffle=True,
                                                                                     transform=transform,
                                                                                     save_clf_data=True,
                                                                                     few_shot=True)
# only original training dataset is used as the whole dataset of the few-shot learning task, so 200 classes in total,
# and in this few-shot learning task's example, 120 classes as training dataset, 40 classes as validation dataset and 40 classes as testing dataset, with 500 images for each class.
```




## Here is a random joke that'll make you laugh!
![Jokes Card](https://readme-jokes.vercel.app/api)