# MiniTorch
MiniTorch is a minimalistic machine learning framework that replicates core functionalities of PyTorch, including tensor operations, auto-differentiation, and neural network modules.

This project is created based on the teaching library developed by Sasha Ruch for the course Machine Learning Engineering at Cornell Tech. Please refer to the documentation [here](https://minitorch.github.io/) for more details about the framework.

## Installation

#### Python Version
MiniTorch requires Python 3.11 or later. Check your version of Python by running `python --version`.

#### Virtual Environment
Set up a virtual environment for this project can help us install packages that are used for this project without impacting the rest of the system. I used `venv` to manage the virtual environment for this project.

First create a workspace for this project:
```
mkdir workspace
```

Then cd into the workspace and create the virtual environment by running the following command:
```
python -m venv .venv
```

To activate the virtual environment, run the following command:
```
source .venv/bin/activate
```

#### Install Packages
When the virtual environment is activated, go to the project folder and install the packages used for this project by running the following commands:
```
python -m pip install -r requirements.txt
python -m pip install -r requirements.extra.txt
```

#### Check if Everything Works
To make sure everything works, run the following command:
```
import minitorch
```

## Start the App
You can stand up the App and start training models and view visualization results by running the following command:

```
streamlit run app.py -- 1
```


## Binary Classification Tasks Training Log Using MiniTorch

### Simple Dataset

#### Model_S_1
* Number of points: 50
* Size of hidden layer: 2
* Learning rate: 0.05
* Number of epochs: 500

Model_S_1 gave a pretty good classification result. The Loss Graph was steadily decreasing.

<img src="/assets/images/simple_1.png" width="50%">
<img src="/assets/images/simple_1_graph.png" width="50%">

#### Model_S_2
* Number of points: 100
* Size of hidden layer: 2
* Learning rate: 0.01
* Number of epochs: 500

Compared to Model_S_1, I only increased the number of points and reduced the learning rate from 0.05 to 0.01. It appears that the learning rate is too small and the Loss Graph is almost flat.

<img src="/assets/images/simple_2.png" width="50%">
<img src="/assets/images/simple_2_graph.png" width="50%">

#### Model_S_3
* Number of points: 100
* Size of hidden layer: 3
* Learning rate: 0.1
* Number of epochs: 500

This model so far gave the best classification result out of the three models. I learned from the previous experiments and increased the learning rate to 0.1. I also bumped the side of hidden layer up by 1 to add more flexibility. This seemed to have worked well.

<img src="/assets/images/simple_3.png" width="50%">
<img src="/assets/images/simple_3_graph.png" width="50%">

### Diag Dataset

#### Model_D_1
* Number of points: 50
* Size of hidden layer: 2
* Learning rate: 0.05
* Number of epochs: 500

For Model_D_1, I used the same initializing parameters as in Model_S_1, and this model provided a fair result correctly classifying the regions, with a little offset on the x_1 and x_2 ratios.

<img src="/assets/images/diag_1.png" width="50%">
<img src="/assets/images/diag_1_graph.png" width="50%">

#### Model_D_2
* Number of points: 150
* Size of hidden layer: 2
* Learning rate: 0.05
* Number of epochs: 500

Based on the result of Model_D_1, I think I should increase the number of data points to the maxixum allowed to achieve a more accurate ratio of the two boundaries, because in this dataset we have unbalanced classes and red points are very few. The experiment confirmed that hypothesis that more data points with the same parameters produced a better classification boundary.

<img src="/assets/images/diag_2.png" width="50%">
<img src="/assets/images/diag_2_graph.png" width="50%">

#### Model_D_3
* Number of points: 150
* Size of hidden layer: 3
* Learning rate: 0.1
* Number of epochs: 500

In Model_D_3, I applied the same idea from Model_S_3 and increased the size of hidden layers to 3. This gave us a sharper decision boundary.

<img src="/assets/images/diag_3.png" width="50%">
<img src="/assets/images/diag_3_graph.png" width="50%">

### Split Dataset

#### Model_SL_1
* Number of points: 100
* Size of hidden layer: 2
* Learning rate: 0.1
* Number of epochs: 500

The data points in this dataset are split on the x_1 axis so I started with 100 data points. This initial set of parameters gave a poor classification result with 62% of points correctly classified.

<img src="/assets/images/split_1.png" width="50%">
<img src="/assets/images/split_1_graph.png" width="50%">

#### Model_SL_2
* Number of points: 100
* Size of hidden layer: 3
* Learning rate: 0.05
* Number of epochs: 500

By turning up the size of hidden layers, I started to see a better result. The colors are roughly right, but the boundaries are very vague.

<img src="/assets/images/split_2.png" width="50%">
<img src="/assets/images/split_2_graph.png" width="50%">

#### Model_SL_3
* Number of points: 150
* Size of hidden layer: 5
* Learning rate: 0.1
* Number of epochs: 500

In the third model for the Split dataset, I used the maximum number of data points and increased the size of hidden layers to 5. This model gave a pretty good classification result. I also observed that different initial settings with the same hyperparameters produces very different training results.

<img src="/assets/images/split_3.png" width="50%">
<img src="/assets/images/split_3_graph.png" width="50%">

### Xor Dataset

#### Model_XoR_1
* Number of points: 150
* Size of hidden layer: 3
* Learning rate: 0.1
* Number of epochs: 500

This dataset had more complex boundaries so I started with the hidden size of layers of 3.

<img src="/assets/images/XoR_1.png" width="50%">
<img src="/assets/images/XoR_1_graph.png" width="50%">

#### Model_XoR_2
* Number of points: 150
* Size of hidden layer: 4
* Learning rate: 0.1
* Number of epochs: 500

By increasing the size of hidden layers to 4, I started to see my data points being classified into 4 regions.

<img src="/assets/images/XoR_2.png" width="50%">
<img src="/assets/images/XoR_2_graph.png" width="50%">

#### Model_XoR_3
* Number of points: 150
* Size of hidden layer: 5
* Learning rate: 0.5
* Number of epochs: 500

Before I got the third model, I tried a number of different set of hyperparameters and found that turning up the learning rate could give me the desired shape of boundaries. In my experiments, I also found that using hidden layer size of 4 - 6 produced similar results. The Loss Graph in these models are NOT steadly decreasing and showed spikes.

<img src="/assets/images/XoR_3.png" width="50%">
<img src="/assets/images/XoR_3_graph.png" width="50%">