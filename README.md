## 1. Setup

####  Requirements

------------

- python3.6
- gym
- OpenCV 
- PyTorch
- tensorboardX

## 2. How to Train

Train network with a separate controller(Original model but with LSTM as the forward network):

```
python3 train.py
```

Train network with controller with shared features with ICM:

```
python3 train.py --shared_features
```

## 3. How to Evaluate

Evaluate network with a separate controller (Original model but with LSTM as the forward network):
```
python3 eval.py --name eta-0.2_stack-1_sparse_extrinsic_run1 --number 5734400
```
Evaluate network with controller with shared features with ICM (Our model):
```
python3 eval.py --name eta-0.2_rnn_forward_both_shared_features_stack-1_only_intrinsic_gradients_feat_run3 --number 5734400
```


References
----------

<https://github.com/ctallec/world-models>

<https://github.com/jcwleo/curiosity-driven-exploration-pytorch>

<https://github.com/pathak22/noreward-rl>

