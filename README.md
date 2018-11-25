# Capstone-Brainwave_Controller
### Current Benchmark (TRY TO IMPROVE IT!):
- SVM Hyperparameter Value:
	SEGMENT_SIZE:3
     Kernel:'rbf'
	C:30
- training_accuracy 0.9873031995937024
- test_accuracy: 0.983756345177665
- test F1 score (each class): [0.9978678  0.98168498 0.97505669 0.98054475]
- test F1 score (weigted): 0.9837477367311879

TODO:
1. Try to improve the overral training accuracy and F1 score.
     -  Hyperparamter tuning (SEGMENT_SIZE, model related parameters, etc.)
     -  Try other classifiers (Neural Networks(MLP, 1-D CNN, RNN), Decision Tree, etc.)
2. Try to use less features to achieve resonable accuracy (PCA).

- Mapping: stop -> 0, left ->1, right -> 2, forward -> 3. (We do not need "backward" for now because this can be achieved by turn left or right twice and go forward)
- I used Muse 2014 and MacOS for code development.

### Setup:
1. Download and install [Muse Developer tools](http://developer.choosemuse.com/tools/mac-tools/getting-started-for-mac)
2. Download Python
3. Connect Muse to your laptop using Bluetooth

### Gather EEG data:
1. Open one terminal and bridge Muse data to your localhost

```muse-io --device Muse --osc osc.udp://127.0.0.1:5000```

2. Open a new terminal, activate Python, and run the server script

```python server.py```

3. Hit ```Ctrl + C``` to stop the server; the CSV files should be created under folder ```Data```
