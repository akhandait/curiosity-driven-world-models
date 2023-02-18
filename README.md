## Our model:

Trained without any extrinsic reward

![Evaluation](https://i.imgur.com/AJrZBIq.gif)

## Architecture:

![Architecture](https://i.imgur.com/K5GUQs8.png)

Please check the [report](https://drive.google.com/file/d/19tvDESmAZK8G8XrFuQilsY962Km1-jZZ/view?usp=sharing) for details of this work.

## Requirements

- python3
- gym
- gym-super-mario-bros
- OpenCV
- PyTorch
- tensorboardX

## Train

Train network with a separate controller(Original model but with LSTM as the forward network):

```
python3 train.py
```

Train network with controller with shared features with ICM (Our model):

```
python3 train.py --shared_features
```

## Evaluate

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

Code has been heavily borrowed from the first two. Thanks a lot!

<https://github.com/ctallec/world-models>

<https://github.com/jcwleo/curiosity-driven-exploration-pytorch>

<https://github.com/pathak22/noreward-rl>

