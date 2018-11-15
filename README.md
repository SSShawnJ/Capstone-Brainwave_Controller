# Capstone-Brainwave_Controller
- Mapping: stop -> 0, left ->1, right -> 2, forward -> 3. (We do not need "backward" for now because this can be achieved by turn left or right twice and go forward)
- I used Muse 2014 and MacOS for code development.



### Setup:
1. Download and install [Muse Developer tools](http://developer.choosemuse.com/tools/mac-tools/getting-started-for-mac)
2. Download Python
3. Connect Muse to my laptop using Bluetooth

### Gather EEG data:
1. Open one terminal and bridge Muse data to my laptop

```muse-io --device Muse --osc osc.udp://127.0.0.1:5000```

2. Open a new terminal, activate Python, and run the server script

```python server.py```

3. Hit ```Ctrl + C``` to stop the server; the CSV files should be created under folder ```Data```
