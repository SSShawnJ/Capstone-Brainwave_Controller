# Capstone-Brainwave_Controller
- I used Muse 2014

### Setup:
1. Download and install [Muse Developer tools](http://developer.choosemuse.com/tools/mac-tools/getting-started-for-mac)
2. Download Python
3. Connect Muse to my laptop using Bluetooth
4. Open one terminal and bridge Muse data to my laptop

```muse-io --device Muse --osc osc.udp://127.0.0.1:5000```

5. Open a new terminal, activate Python, and run the server script

```python server.py```
