# Diabetic-Retinopathy-Classification
Multi-Class Classification of retina images using deep learning

Take a look to my Medium Post about this project [here](https://medium.com/@aleperez119/using-a-resnet50-model-to-classify-retinopathy-images-e95968ab69f0)!

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)

## Installation <a name="installation"></a>

Use the Python version 3.7.6 and install the libraries in the requirements.txt file of this repo

1. Download the dataset from [this link](https://www.kaggle.com/datasets/amanneo/diabetic-retinopathy-resized-arranged). Then, unzip the compressed file.

2. Download the trained model from: [link](https://drive.google.com/file/d/1BCtLviqhYIt2782qGrnZv67v7smMQv7z/view?usp=sharing)

Your project structure should be like this:

``` bash 
    .
    ├── data.ipynb
    ├── retinopathy-model-adam-ce.pt
    ├── archive       
    │   ├── 0 
    │   │   └── *.jpeg        
    │   ...
    │   └── 4 
    │       └── *.jpeg 
    ├── utils       
    │   ├── dataprocessing.py
    │   ├── engine.py
    │   ├── visualizations.py
    └── ...
```

## Project Motivation<a name="motivation"></a>

The aim of this project is to classify retina images into different types of Diabetic Retinopathy using deep learning, specifically a ResNet50 convolutional neural network. Diabetic Retinopathy is a common complication of diabetes and can lead to blindness if left untreated. Early detection of the condition is crucial to prevent blindness. Hence, it is important to train models able to diagnose these pathologies.

## File Descriptions <a name="files"></a>

This repo is organized this way:
- The archive folder where the dataset is stored.
- The utils folder that contains the dataprocessing, engine and visualization scripts that hold functions that are used in the jupyter notebook.
- Tha jupyter notebook, data.ipynb where data exploration, visualizations, training, and evaluation is made.

## Results<a name="results"></a>

### Model Performance Metrics 

``` bash 
|                    | **precision** | **recall** | **f1-score** |
|-------------------:|--------------:|-----------:|-------------:|
|                  0 |        0.7676 |     0.9504 |       0.8493 |
|                  1 |        0.0000 |     0.0000 |       0.0000 |
|                  2 |        0.3211 |     0.1076 |       0.1612 |
|                  3 |        0.1036 |     0.1657 |       0.1275 |
|                  4 |        0.0000 |     0.0000 |       0.0000 |
|       **accuracy** |        0.7185 |     0.7185 |       0.7185 |
|      **macro avg** |        0.2385 |     0.2448 |       0.2276 |
| **weighted   avg** |        0.6149 |     0.7185 |       0.6514 |
```



## Instructions <a name="instructions"></a>
1. After downloading the dataset and the model weigths as in the [Installation Section](#installation), run the jupyter notebook cells. 

## Acknowledgements<a name="acknowledgements"></a>

Credits to Kaggle for the data. The code of this project was based on [this notebook](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb#scrollTo=hupBoXNhbqe_)
