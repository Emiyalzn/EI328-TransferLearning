# README

This file introduce the usage and outline of handin.

The handin project has the following directory structure

~~~
transfer_learning
|-- README.md # this file
|-- checkpoint # subdir to save checkpoint of model or components of model
|-- code        # the src code of the prj
|   |-- data_augmentation.py # for data augmentation
|   |-- dataset.py # for load train/test data
|   |-- main.py # pipeline of each model
|   |-- models.py # each model class
|   |-- parser.py # args definition
|   |-- utils.py # utils for data visualization
`-- dataset # SEED dataset
~~~

## How to run

To run the baseline and our model, go to path /transfer_learning/code, run command.

~~~shell
python main.py --model xx
~~~

the model arg have the following selection:

- Conventional ML model:
  - SVM
- Conventional DL models:
  - MLP
  - ResNet
- Domain generalization models:
  - IRM
  - REx
- Domain adaptation models:
  - DANN
  - ADDA
  - SADA (ours)
- Data augmentation:
  - WGANGen

More args can be set to tune the model, the details is in the file parser.py.

To conduct data augmentation and run on WGANGen, run command

~~~shell
python data_augmentation --model WGANGen
~~~

You can also use args like `--lr` or `--batch_size` to tune the hyperparameters.

## Experiment Setup

All experiments are done on **i9-10920X CPU @3.50GHz** along with a **GeForce RTX 3090 GPU**.

The software environment is **pytorch1.8/CUDA11.0/sklearn**.
