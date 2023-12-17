# Inception-V2 on CIFAR-10

This folder the code for the experiments we conducted to apply the Net2Net techniques to the Inception-V2 architecture on the CIFAR-10 dataset.

It is organized as follows:
- `main_wider.py` contains the code to run the experiments with the Net2WiderNet technique.
- `main_deeper.py` contains the code to run the experiments with the Net2DeeperNet technique.
- `demo_wider.ipynb` is a Jupyter notebook that can be used to visualize the principle of the Net2WiderNet technique.
- `demo_net2deeper.ipynb` is a Jupyter notebook that can be used to visualize the principle of the Net2DeeperNet technique.
- `tune_learning_rate.py` contains the code to tune the learning rate of the models used to evaluate the Net2WiderNet technique.
- `logs/` is a folder used to store the logs of the experiments.