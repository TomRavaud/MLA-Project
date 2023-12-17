## Logs

This folder contains the log files generated after each NN training and testing.

The log files are organized in a subfolder as follows:

```bash
{log_directory_name}/
├── accuracy_curve.png
├── network.params
├── network.txt
├── parameters_table.txt
├── test_results.txt
├── train_accuracies.npy
└── val_accuracies.npy
```

where:
- `accuracy_curve.png` is the plot of the training and validation accuracies as a function of the training epochs.
- `network.params` is the file containing the parameters of the trained network.
- `network.txt` is the file containing the architecture of the trained network.
- `parameters_table.txt` is the file containing the parameters of the experiment.
- `test_results.txt` is the file containing the test results of the trained network.
- `train_accuracies.npy` is the file containing the training accuracies as a function of the training epochs.
- `val_accuracies.npy` is the file containing the validation accuracies as a function of the training epochs.