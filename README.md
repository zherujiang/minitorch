# MiniTorch
MiniTorch is a minimalistic machine learning framework that replicates core functionalities of PyTorch, including tensor operations, auto-differentiation, and neural network modules.

This project is created based on the teaching library developed by Sasha Ruch for the course Machine Learning Engineering at Cornell Tech. Please refer to the documentation [here](https://minitorch.github.io/) for more details about the framework.

## Installation

### Python Version
MiniTorch requires Python 3.11 or later. Check your version of Python by running `python --version`.

### Virtual Environment
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

### Install Packages
When the virtual environment is activated, go to the directory containing this project and install the packages used for this project by running the following commands:
```
python -m pip install -r requirements.txt
python -m pip install -r requirements.extra.txt
```

Be sure to also install minitorch in your Virtual Env.
```
pip install -Ue .
```

### Check if Everything Works
To make sure everything works, run the following command in `python`:
```
import minitorch
```

## Start the App
Enter the `minitorch/project` directory. You can stand up the App and start training models and view visualization results by running the following command:

```
streamlit run app.py -- {module_number}
```
### App Preview
<img src="/assets/images/app_training_1.png" width="100%">
<img src="/assets/images/app_training_2.png" width="100%">
<img src="/assets/images/app_training_3.png" width="100%">
<img src="/assets/images/app_training_4.png" width="100%">

<img src="/assets/images/app_tensor_sandbox_1.png" width="100%">
<img src="/assets/images/app_tensor_sandbox_2.png" width="100%">
<img src="/assets/images/app_math_function_sandbox.png" width="100%">
<img src="/assets/images/app_autogradient_sandbox.png" width="100%">

## Binary Classification Tasks Training Example Using MiniTorch

### Simple Dataset

#### Model Parameters
* Number of points: 150
* Size of hidden layer: 2
* Learning rate: 0.05
* Number of epochs: 500

#### Training Result
* Time per epoch: 0.089s
* Correct: 150/150
* Loss: 32.5625

<img src="/assets/images/simple.png" width="50%">
<img src="/assets/images/simple_loss.png" width="50%">


### Diag Dataset
#### Model Parameters
* Number of points: 150
* Size of hidden layer: 3
* Learning rate: 0.1
* Number of epochs: 500

#### Training Result
* Time per epoch: 0.133s
* Correct: 148/150
* Loss: 11.7703

<img src="/assets/images/diag.png" width="50%">
<img src="/assets/images/diag_loss.png" width="50%">


### Split Dataset

#### Model Parameters
* Number of points: 150
* Size of hidden layer: 4
* Learning rate: 0.1
* Number of epochs: 500

#### Training Result
* Time per epoch: 0.209s
* Correct: 146/150
* Loss: 36.9434

<img src="/assets/images/split.png" width="50%">
<img src="/assets/images/split_loss.png" width="50%">


### XoR Dataset
#### Model Parameters
* Number of points: 138
* Size of hidden layer: 6
* Learning rate: 0.1
* Number of epochs: 500

#### Training Result
* Time per epoch: 0.376s
* Correct: 130/138
* Loss: 44.8873

<img src="/assets/images/xor.png" width="50%">
<img src="/assets/images/xor_loss.png" width="50%">

