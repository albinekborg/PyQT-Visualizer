# Visualizer
This is a real-time audio visualizer built with PyQt5, PyAudio, and pyqtgraph. It creates a dynamic spectrogram that reacts to sound input from your microphone, displaying a frequency-domain transformation of the sound. The visualizer also includes customizable themes using JSON. 

## Screenshot
![image](image.png "Image")

## Requirements
To run the visualizer, you need the following Python packages:

```bash
pip install numpy pyaudio PyQt5 pyqtgraph matplotlib
```
## How to run
1. Clone or download this repository.
2. Install the required packages.
3. Run the visualizer using:

```bash
python visualizer.py [theme_name]
```

If no theme is specified, it defaults to "summer" theme.
