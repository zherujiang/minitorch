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

# Training Log Using MiniTorch

## Simple Dataset Binary Classification
### CPU Backend (Parallel Optimization using Numba) Training
#### Model Parameters
* Number of points: 150
* Size of hidden layer: 100
* Learning rate: 0.05
* Number of epochs: 200
* Batch size: 50

#### Training Result
* Time per epoch: 0.091s
* Correct: 148/150
* Loss: 2.6133

<img src="/assets/images/simple_cpu.png" width="50%">
<img src="/assets/images/simple_loss_cpu.png" width="50%">
<img src="/assets/images/simple_log_cpu.png" width="50%">

### GPU Backend (CudaOps) Training 
#### Model Parameters
* Number of points: 150
* Size of hidden layer: 100
* Learning rate: 0.05
* Number of epochs: 500
* Batch size: 50

#### Training Result
* Time per epoch: 0.797s
* Correct: 150/150
* Loss: 1.3223
```
Epoch  10  loss  13.262209187553221 correct 136
Epoch  20  loss  9.416668025095028 correct 141
Epoch  30  loss  10.171094173141213 correct 143
Epoch  40  loss  6.144273666880221 correct 145
Epoch  50  loss  6.338275444197706 correct 147
Epoch  60  loss  5.315298202614823 correct 147
Epoch  70  loss  6.3532484481758456 correct 147
Epoch  80  loss  4.297244308942781 correct 148
Epoch  90  loss  6.131539233745611 correct 148
Epoch  100  loss  3.976840177208355 correct 148
Epoch  110  loss  3.0373080682170217 correct 149
Epoch  120  loss  4.314400773016098 correct 149
Epoch  130  loss  3.511654245270693 correct 149
Epoch  140  loss  4.704098435333334 correct 149
Epoch  150  loss  4.84005991537246 correct 149
Epoch  160  loss  3.2546175690645898 correct 149
Epoch  170  loss  3.0566685025653535 correct 149
Epoch  180  loss  3.997642634298782 correct 150
Epoch  190  loss  3.3274577664029414 correct 148
Epoch  200  loss  3.962327843114556 correct 148
Epoch  210  loss  5.161806310102181 correct 149
Epoch  220  loss  2.1079047836632614 correct 150
Epoch  230  loss  2.177955220083965 correct 150
Epoch  240  loss  3.9062136780463437 correct 148
Epoch  250  loss  3.1906409548167436 correct 150
Epoch  260  loss  2.5284227437670888 correct 149
Epoch  270  loss  2.2034935666005193 correct 150
Epoch  280  loss  4.243802052327457 correct 149
Epoch  290  loss  1.7189281843499877 correct 150
Epoch  300  loss  2.6690208484129507 correct 150
Epoch  310  loss  2.246076003943947 correct 150
Epoch  320  loss  3.3071195804550184 correct 149
Epoch  330  loss  2.5360937486645563 correct 150
Epoch  340  loss  2.203562190599043 correct 149
Epoch  350  loss  2.573090158122847 correct 150
Epoch  360  loss  2.762776240187227 correct 150
Epoch  370  loss  3.3576132997094033 correct 150
Epoch  380  loss  2.0554648428420315 correct 150
Epoch  390  loss  2.6279684758112096 correct 150
Epoch  400  loss  2.5097504659787084 correct 150
Epoch  410  loss  2.2144838926962986 correct 150
Epoch  420  loss  1.9180491355895386 correct 150
Epoch  430  loss  1.9297033460476634 correct 149
Epoch  440  loss  2.016608431141371 correct 150
Epoch  450  loss  1.935358596677633 correct 150
Epoch  460  loss  1.1970807599073305 correct 150
Epoch  470  loss  0.9662017141560335 correct 150
Epoch  480  loss  1.4550626708868193 correct 150
Epoch  490  loss  1.6711099939183105 correct 149
Epoch  500  loss  1.3223574739322064 correct 150
Time per epoch:  0.7969081153869629s
```

## Split Dataset Binary Classification
### CPU Backend (Parallel Optimization using Numba) Training
#### Model Parameters
* Number of points: 150
* Size of hidden layer: 100
* Learning rate: 0.05
* Number of epochs: 200
* Batch size: 50

#### Training Result
* Time per epoch: 0.100s
* Correct: 148/150
* Loss: 7.2633

<img src="/assets/images/split_cpu.png" width="50%">
<img src="/assets/images/split_loss_cpu.png" width="50%">
<img src="/assets/images/split_log_cpu.png" width="50%">

### GPU Backend (CudaOps) Training
#### Model Parameters
* Number of points: 100
* Size of hidden layer: 100
* Learning rate: 0.05
* Number of epochs: 500
* Batch size: 50

#### Training Result
* Time per epoch: 0.521s
* Correct: 99/100
* Loss: 1.7115

```
Epoch  10  loss  12.434918714612413 correct 96
Epoch  20  loss  10.987760323938975 correct 96
Epoch  30  loss  10.583358455974302 correct 96
Epoch  40  loss  7.702080017033701 correct 96
Epoch  50  loss  6.415706986936865 correct 96
Epoch  60  loss  5.901910041371588 correct 98
Epoch  70  loss  4.615673023619884 correct 98
Epoch  80  loss  4.940653103376711 correct 98
Epoch  90  loss  5.6404829454739644 correct 98
Epoch  100  loss  4.424901680934736 correct 98
Epoch  110  loss  4.254901485659172 correct 98
Epoch  120  loss  5.3880779597945025 correct 99
Epoch  130  loss  4.774886572935284 correct 99
Epoch  140  loss  3.433912606183646 correct 99
Epoch  150  loss  3.0432262040399407 correct 99
Epoch  160  loss  2.7896922698085795 correct 99
Epoch  170  loss  4.827306587921807 correct 99
Epoch  180  loss  2.4078879382507674 correct 99
Epoch  190  loss  3.6203354068190716 correct 99
Epoch  200  loss  4.396458886365748 correct 99
Epoch  210  loss  2.0450155315294953 correct 99
Epoch  220  loss  2.884373428278798 correct 99
Epoch  230  loss  1.960623952698316 correct 99
Epoch  240  loss  2.0590238761776103 correct 99
Epoch  250  loss  3.1212010650980035 correct 99
Epoch  260  loss  1.3189152664715462 correct 99
Epoch  270  loss  2.7127620319746337 correct 99
Epoch  280  loss  2.4136157746858666 correct 99
Epoch  290  loss  2.205345407926015 correct 99
Epoch  300  loss  3.41613665954947 correct 99
Epoch  310  loss  2.022091964407795 correct 99
Epoch  320  loss  2.747145088493211 correct 99
Epoch  330  loss  2.8832100717296143 correct 99
Epoch  340  loss  2.0166800981869346 correct 100
Epoch  350  loss  2.019569205183759 correct 99
Epoch  360  loss  3.245031925279889 correct 100
Epoch  370  loss  2.725312784468276 correct 99
Epoch  380  loss  1.5520643033984673 correct 99
Epoch  390  loss  1.0528778075552534 correct 99
Epoch  400  loss  2.4574409166555355 correct 99
Epoch  410  loss  1.3199850379722 correct 100
Epoch  420  loss  1.7975769861485655 correct 99
Epoch  430  loss  1.6731806518116712 correct 99
Epoch  440  loss  1.2450589521721094 correct 99
Epoch  450  loss  0.9965329813089921 correct 100
Epoch  460  loss  1.350060232037304 correct 99
Epoch  470  loss  2.3731953451821033 correct 100
Epoch  480  loss  1.7422895471244024 correct 100
Epoch  490  loss  2.035818443717936 correct 100
Epoch  500  loss  1.711487965220122 correct 99
Time per epoch:  0.5208355765342713s
```

## XOR Dataset Binary Classification
### CPU Backend (Parallel Opetimization using Numba) Training
#### Model Parameters
* Number of points: 150
* Size of hidden layer: 100
* Learning rate: 0.05
* Number of epochs: 300
* Batch size: 50

#### Training Result
* Time per epoch: 0.087s
* Correct: 144/150
* Loss: 8.2335
<img src="/assets/images/XoR_cpu.png" width="50%">
<img src="/assets/images/XoR_loss_cpu.png" width="50%">
<img src="/assets/images/XoR_log_cpu.png" width="50%">

### GPU Backend (CudaOps) Training
#### Model Parameters
* Number of points: 100
* Size of hidden layer: 100
* Learning rate: 0.05
* Number of epochs: 500
* Batch size: 50

#### Training Result
* Time per epoch: 0.523s
* Correct: 97/100
* Loss: 4.8475
```
Epoch  10  loss  24.8519267355222 correct 88
Epoch  20  loss  21.890465997652683 correct 88
Epoch  30  loss  20.027433172651516 correct 88
Epoch  40  loss  20.476473655912834 correct 90
Epoch  50  loss  16.779492547341906 correct 92
Epoch  60  loss  14.673046529985623 correct 92
Epoch  70  loss  16.312974902607998 correct 90
Epoch  80  loss  13.502664213366442 correct 92
Epoch  90  loss  15.21449017799693 correct 94
Epoch  100  loss  12.915720463019337 correct 94
Epoch  110  loss  12.293044591740935 correct 92
Epoch  120  loss  10.215260215600084 correct 94
Epoch  130  loss  11.03009857522236 correct 94
Epoch  140  loss  10.019816149494382 correct 94
Epoch  150  loss  11.323382152724584 correct 94
Epoch  160  loss  11.049438765590981 correct 94
Epoch  170  loss  10.10388014679021 correct 94
Epoch  180  loss  7.547354872388958 correct 94
Epoch  190  loss  10.057805546880182 correct 94
Epoch  200  loss  7.609268474399949 correct 94
Epoch  210  loss  7.9075112662500295 correct 95
Epoch  220  loss  10.230965928654909 correct 94
Epoch  230  loss  8.612783867617498 correct 94
Epoch  240  loss  7.66527779691888 correct 94
Epoch  250  loss  7.643342885427886 correct 94
Epoch  260  loss  7.946705942205567 correct 94
Epoch  270  loss  6.048573647707716 correct 94
Epoch  280  loss  7.109607103010665 correct 94
Epoch  290  loss  9.09455283686561 correct 95
Epoch  300  loss  6.1846439906509705 correct 94
Epoch  310  loss  6.6426469169610085 correct 95
Epoch  320  loss  8.131495195168977 correct 95
Epoch  330  loss  7.628988047172486 correct 94
Epoch  340  loss  7.5108856957739265 correct 95
Epoch  350  loss  3.850822007362755 correct 95
Epoch  360  loss  7.418223672648361 correct 96
Epoch  370  loss  4.367012114183443 correct 95
Epoch  380  loss  8.052063013510095 correct 97
Epoch  390  loss  7.079680357739138 correct 95
Epoch  400  loss  4.687085013110687 correct 96
Epoch  410  loss  6.387848023271869 correct 96
Epoch  420  loss  6.73205660878836 correct 96
Epoch  430  loss  6.712719442087073 correct 98
Epoch  440  loss  4.168085557982674 correct 97
Epoch  450  loss  7.702824431935882 correct 97
Epoch  460  loss  4.121436573500693 correct 97
Epoch  470  loss  5.875229435058945 correct 97
Epoch  480  loss  6.106633519056049 correct 97
Epoch  490  loss  6.9322980173625615 correct 97
Epoch  500  loss  4.847461218015787 correct 97
Time per epoch:  0.5225029020309448s
```

## Big Model - XOR Dataset with 200 Hidden Size
### CPU Backend (Parallel Optimization using Numba) Training
#### Model Parameters
* Number of points: 150
* Size of hidden layer: 200
* Learning rate: 0.05
* Number of epochs: 300
* Batch size: 50

#### Training Result
* Time per epoch: 0.160s
* Correct: 142/150
* Loss: 7.7792
<img src="/assets/images/XoR_big_cpu.png" width="50%">
<img src="/assets/images/XoR_big_loss_cpu.png" width="50%">
<img src="/assets/images/XoR_big_log_cpu.png" width="50%">

### GPU Backend (CudaOps) Training
#### Model Parameters
* Number of points: 150
* Size of hidden layer: 200
* Learning rate: 0.05
* Number of epochs: 500
* Batch size: 50

#### Training Result
* Time per epoch: 0.852s
* Correct: 150/150
* Loss: 3.1303

```
Epoch  10  loss  42.0072417263013 correct 83
Epoch  20  loss  18.66098879104596 correct 139
Epoch  30  loss  15.495114987103527 correct 141
Epoch  40  loss  14.583600550736307 correct 123
Epoch  50  loss  13.469071855080186 correct 141
Epoch  60  loss  8.13507504988055 correct 145
Epoch  70  loss  14.631315111787284 correct 133
Epoch  80  loss  9.285737692766649 correct 148
Epoch  90  loss  7.528064236851376 correct 148
Epoch  100  loss  9.426732082873677 correct 142
Epoch  110  loss  9.216520583543705 correct 148
Epoch  120  loss  7.733500254751689 correct 149
Epoch  130  loss  5.515715741567107 correct 149
Epoch  140  loss  7.959565329383627 correct 146
Epoch  150  loss  6.189307372627531 correct 147
Epoch  160  loss  6.769136499414869 correct 149
Epoch  170  loss  5.777736022868717 correct 149
Epoch  180  loss  6.5296982828498225 correct 150
Epoch  190  loss  6.447426150117004 correct 149
Epoch  200  loss  5.21692924152071 correct 150
Epoch  210  loss  3.6286625257828153 correct 150
Epoch  220  loss  4.10152367014177 correct 146
Epoch  230  loss  4.001968014589115 correct 148
Epoch  240  loss  4.299465360109346 correct 150
Epoch  250  loss  4.158609553659765 correct 150
Epoch  260  loss  4.059390789408351 correct 149
Epoch  270  loss  3.8120059513064373 correct 149
Epoch  280  loss  3.057564132479502 correct 150
Epoch  290  loss  2.8559804006607665 correct 150
Epoch  300  loss  6.036910569022698 correct 146
Epoch  310  loss  3.226612234972243 correct 150
Epoch  320  loss  4.149033898160834 correct 150
Epoch  330  loss  3.451076342226277 correct 150
Epoch  340  loss  3.705906543748794 correct 148
Epoch  350  loss  1.8661210895886091 correct 150
Epoch  360  loss  3.7310478150911326 correct 150
Epoch  370  loss  3.238220219204665 correct 150
Epoch  380  loss  3.303285342110425 correct 150
Epoch  390  loss  3.6360060996646117 correct 150
Epoch  400  loss  2.158255974289149 correct 150
Epoch  410  loss  2.225696676221485 correct 150
Epoch  420  loss  2.326073034048486 correct 150
Epoch  430  loss  2.1545277388613253 correct 150
Epoch  440  loss  2.673806788408854 correct 150
Epoch  450  loss  3.5407404902362463 correct 150
Epoch  460  loss  2.3494470366197784 correct 150
Epoch  470  loss  2.587967241452305 correct 150
Epoch  480  loss  2.146729363327209 correct 150
Epoch  490  loss  1.5180580151845975 correct 150
Epoch  500  loss  3.130340612629343 correct 150
Time per epoch: 0.8519989128112793s
```

# Matrix Multiplication Speedup Test
## Runtime Output
```
Training Time summary (s)
Matrix Size: 64
    cpu(FastOps): 0.00312
    gpu(CudaOps): 0.00568
Matrix Size: 128
    cpu(FastOps): 0.01575
    gpu(CudaOps): 0.01575
Matrix Size: 256
    cpu(FastOps): 0.09136
    gpu(CudaOps): 0.04460
Matrix Size: 512
    cpu(FastOps): 0.99209
    gpu(CudaOps): 0.18844
Matrix Size: 1024
    cpu(FastOps): 7.99234
    gpu(CudaOps): 1.00060
```
<img src="/assets/images/matmul_speedtest_graph.png" width="80%">


# Parallel Optimization Diagnostic outputs

Run the following parallel analytics script to get the diagnostic outputs:
```
python project/parallel_check.py
```

```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/zhe
rujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (174)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/zherujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-zherujiang/minitorch/fast_ops.py (174) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        # Task 3.1                                                           | 
        size = len(out)                                                      | 
                                                                             | 
        if np.array_equal(out_strides, in_strides) and np.array_equal(       | 
            out_shape, in_shape                                              | 
        ):                                                                   | 
            # Parallel loop over all elements                                | 
            for i in prange(size):-------------------------------------------| #2
                out[i] += fn(in_storage[i])                                  | 
        else:                                                                | 
            # Parallel loop over all elements                                | 
            for i in prange(size):-------------------------------------------| #3
                # Convert ordinal index to n-dimensional index               | 
                out_index = np.zeros(len(out_shape), np.int32)---------------| #0
                in_index = np.zeros(len(in_shape), np.int32)-----------------| #1
                                                                             | 
                # Map output index to input index (broadcasting)             | 
                to_index(i, out_shape, out_index)                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                                                                             | 
                # Calculate positions in storage                             | 
                out_pos = index_to_position(out_index, out_strides)          | 
                in_pos = index_to_position(in_index, in_strides)             | 
                                                                             | 
                # Apply the function and store result in a local copy        | 
                out[out_pos] += fn(in_storage[in_pos])                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/zherujiang/myDrive/
Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (195) is hoisted out of the parallel loop 
labelled #3 (it will be performed before the loop is executed and reused inside 
the loop):
   Allocation:: out_index = np.zeros(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/zherujiang/myDrive/
Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (196) is hoisted out of the parallel loop 
labelled #3 (it will be performed before the loop is executed and reused inside 
the loop):
   Allocation:: in_index = np.zeros(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/zhe
rujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (235)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/zherujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-zherujiang/minitorch/fast_ops.py (235) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # Task 3.1.                                                        | 
        size = len(out)                                                    | 
                                                                           | 
        if (                                                               | 
            np.array_equal(out_strides, a_strides)                         | 
            and np.array_equal(out_strides, b_strides)                     | 
            and np.array_equal(out_shape, a_shape)                         | 
            and np.array_equal(out_shape, b_shape)                         | 
        ):                                                                 | 
            # Parallel loop over all elements                              | 
            for i in prange(size):-----------------------------------------| #7
                out[i] += fn(a_storage[i], b_storage[i])                   | 
        else:                                                              | 
            # Parallel loop over all elements                              | 
            for i in prange(size):-----------------------------------------| #8
                # Convert ordinal index to n-dimensional index             | 
                out_index = np.zeros(len(out_shape), np.int32)-------------| #4
                a_index = np.zeros(len(a_shape), np.int32)-----------------| #5
                b_index = np.zeros(len(b_shape), np.int32)-----------------| #6
                                                                           | 
                # Map output index to input index (broadcasting)           | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                                                                           | 
                # Calculate positions in storage                           | 
                out_pos = index_to_position(out_index, out_strides)        | 
                a_pos = index_to_position(a_index, a_strides)              | 
                b_pos = index_to_position(b_index, b_strides)              | 
                                                                           | 
                # Apply the function and store result in a local copy      | 
                out[out_pos] += fn(a_storage[a_pos], b_storage[b_pos])     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4, #5, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial)
   +--6 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/zherujiang/myDrive/
Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (262) is hoisted out of the parallel loop 
labelled #8 (it will be performed before the loop is executed and reused inside 
the loop):
   Allocation:: out_index = np.zeros(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/zherujiang/myDrive/
Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (263) is hoisted out of the parallel loop 
labelled #8 (it will be performed before the loop is executed and reused inside 
the loop):
   Allocation:: a_index = np.zeros(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/zherujiang/myDrive/
Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (264) is hoisted out of the parallel loop 
labelled #8 (it will be performed before the loop is executed and reused inside 
the loop):
   Allocation:: b_index = np.zeros(len(b_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/zherujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (303)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/zherujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-zherujiang/minitorch/fast_ops.py (303) 
--------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                  | 
        out: Storage,                                                                             | 
        out_shape: Shape,                                                                         | 
        out_strides: Strides,                                                                     | 
        a_storage: Storage,                                                                       | 
        a_shape: Shape,                                                                           | 
        a_strides: Strides,                                                                       | 
        reduce_dim: int,                                                                          | 
    ) -> None:                                                                                    | 
        # Task 3.1                                                                                | 
        # Calculate the size of the output and the dimension to reduce                            | 
        size = len(out)                                                                           | 
        reduce_size = a_shape[reduce_dim]                                                         | 
                                                                                                  | 
        # Parallel loop over all output elements                                                  | 
        for i in prange(size):--------------------------------------------------------------------| #10
            # Convert output index to n-dimensional index                                         | 
            out_index = np.zeros(len(out_shape), np.int32)----------------------------------------| #9
            to_index(i, out_shape, out_index)                                                     | 
                                                                                                  | 
            # Get output position                                                                 | 
            out_pos = index_to_position(out_index, out_strides)                                   | 
                                                                                                  | 
            temp = out[out_pos]                                                                   | 
            # Reduce over the specified dimension, reuse out_index for indexing into a_storage    | 
            for j in range(reduce_size):                                                          | 
                out_index[reduce_dim] = j                                                         | 
                a_pos = index_to_position(out_index, a_strides)                                   | 
                temp = fn(temp, a_storage[a_pos])                                                 | 
                                                                                                  | 
            out[out_pos] = temp                                                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/zherujiang/myDrive/
Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (320) is hoisted out of the parallel loop 
labelled #10 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index = np.zeros(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/zher
ujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-
zherujiang/minitorch/fast_ops.py (338)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/zherujiang/myDrive/Cornell_courses/CS5781_MLE/miniTorch_workspace/mod3-zherujiang/minitorch/fast_ops.py (338) 
----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                  | 
    out: Storage,                                                                             | 
    out_shape: Shape,                                                                         | 
    out_strides: Strides,                                                                     | 
    a_storage: Storage,                                                                       | 
    a_shape: Shape,                                                                           | 
    a_strides: Strides,                                                                       | 
    b_storage: Storage,                                                                       | 
    b_shape: Shape,                                                                           | 
    b_strides: Strides,                                                                       | 
) -> None:                                                                                    | 
    """NUMBA tensor matrix multiply function.                                                 | 
                                                                                              | 
    Should work for any tensor shapes that broadcast as long as                               | 
                                                                                              | 
    ```                                                                                       | 
    assert a_shape[-1] == b_shape[-2]                                                         | 
    ```                                                                                       | 
                                                                                              | 
    Optimizations:                                                                            | 
                                                                                              | 
    * Outer loop in parallel                                                                  | 
    * No index buffers or function calls                                                      | 
    * Inner loop should have no global writes, 1 multiply.                                    | 
                                                                                              | 
                                                                                              | 
    Args:                                                                                     | 
    ----                                                                                      | 
        out (Storage): storage for `out` tensor                                               | 
        out_shape (Shape): shape for `out` tensor                                             | 
        out_strides (Strides): strides for `out` tensor                                       | 
        a_storage (Storage): storage for `a` tensor                                           | 
        a_shape (Shape): shape for `a` tensor                                                 | 
        a_strides (Strides): strides for `a` tensor                                           | 
        b_storage (Storage): storage for `b` tensor                                           | 
        b_shape (Shape): shape for `b` tensor                                                 | 
        b_strides (Strides): strides for `b` tensor                                           | 
                                                                                              | 
    Returns:                                                                                  | 
    -------                                                                                   | 
        None : Fills in `out`                                                                 | 
                                                                                              | 
    """                                                                                       | 
    # Task 3.2                                                                                | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                    | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                    | 
                                                                                              | 
    # Get the dimensions for the matrix multiplication                                        | 
    batch_size = out_shape[0]                                                                 | 
    rows = out_shape[1]                                                                       | 
    cols = out_shape[2]                                                                       | 
    reduce_dim = a_shape[2]                                                                   | 
                                                                                              | 
    for batch in prange(batch_size):----------------------------------------------------------| #11
        for i in range(rows):                                                                 | 
            for j in range(cols):                                                             | 
                a_inner = batch * a_batch_stride + i * a_strides[1]                           | 
                b_inner = batch * b_batch_stride + j * b_strides[2]                           | 
                                                                                              | 
                accum = 0                                                                     | 
                for k in range(reduce_dim):                                                   | 
                    accum += a_storage[a_inner] * b_storage[b_inner]                          | 
                    a_inner += a_strides[2]                                                   | 
                    b_inner += b_strides[1]                                                   | 
                                                                                              | 
                out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]    | 
                out[out_pos] = accum                                                          | 
                                                                                              | 
    # Parallel loop over batches and rows                                                     | 
    # for batch in prange(batch_size):                                                        | 
    #     a_pos_i = batch * a_batch_stride                                                    | 
    #     b_pos_b = batch * b_batch_stride                                                    | 
    #     out_pos_i = batch * out_strides[0]                                                  | 
                                                                                              | 
    #     for i in range(rows):                                                               | 
    #         b_pos_j = b_pos_b                                                               | 
    #         out_pos_j = out_pos_i                                                           | 
    #         for j in range(cols):                                                           | 
    #             # Initialize accumulator                                                    | 
    #             acc = 0.0                                                                   | 
                                                                                              | 
    #             a_pos = a_pos_i                                                             | 
    #             b_pos = b_pos_j                                                             | 
    #             out_pos = out_pos_j                                                         | 
    #             # Inner reduction loop                                                      | 
    #             for k in range(reduce_dim):                                                 | 
    #                 acc += a_storage[a_pos] * b_storage[b_pos]                              | 
    #                 # update a_pos and b_pos                                                | 
    #                 a_pos += a_strides[2]                                                   | 
    #                 b_pos += b_strides[1]                                                   | 
                                                                                              | 
    #             # Store result                                                              | 
    #             out[out_pos] += acc                                                         | 
                                                                                              | 
    #             # update b_pos and out_pos                                                  | 
    #             b_pos_j += b_strides[2]                                                     | 
    #             out_pos_j += out_strides[2]                                                 | 
                                                                                              | 
    #         # update a_pos and out_pos                                                      | 
    #         a_pos_i += a_strides[1]                                                         | 
    #         out_pos_i += out_strides[1]                                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
