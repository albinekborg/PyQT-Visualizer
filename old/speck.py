import sys
import numpy as np
import pyaudio
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# Parameters for the audio stream
CHUNK = 256  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for microphone
RATE = 44100//4  # Samples per second
N = 8

class AudioSpectrogram(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.prev_img = None

        # Timer setup for updating the plot
        self.timer = QTimer()
        self.timer.setInterval(0)  # Milliseconds interval for timer
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.yf = 0
        self.buf_size = 128
        self.buf = np.zeros((self.buf_size, N*CHUNK//2))  # Ensure the buffer is properly shaped
        self.buf_iter = 0

        self.tbuf = np.zeros(CHUNK*N)

    def update_buf(self):
        x = self.yf * np.linspace(0,1,len(self.yf))
        self.buf = np.concatenate([x[np.newaxis,:], self.buf[:-1]], axis=0)
        self.buf *= .90

    def update_tbuf(self, data):
        self.tbuf = np.concatenate([data, self.tbuf[:-CHUNK]])

    def init_ui(self):
        self.setWindowTitle("Live Spectrogram")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)

        colormap = pg.colormap.get('Spectral', source='matplotlib')  # Choosing 'viridis' colormap
        lut = colormap.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(lut)
        self.img_item.setLevels([0, 10])  # Adjust levels based on your data

        self.plot_widget.getPlotItem().getAxis('bottom').setLogMode(False)


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    
    def update(self):
        alpha = 0.75
        # Read audio data and convert it directly from buffer
        data = np.frombuffer(self.stream.read(CHUNK, exception_on_overflow=True), dtype=np.int16)

        self.update_tbuf(data)
        mask = np.hanning(len(self.tbuf))
        masked_tbuf = self.tbuf * mask
        self.yf = np.abs(np.fft.rfft(masked_tbuf))[1:]
        self.update_buf()
        # Display the updated spectrogram buffer

        hann_window_1d_vertical = np.hanning(self.buf.shape[0])[:, np.newaxis]  # vertical Hanning
        hann_window_1d_horizontal = np.hanning(self.buf.shape[1])  # horizontal Hanning
        buf_win = hann_window_1d_vertical * hann_window_1d_horizontal  # outer product to create 2D window


        self.img = np.abs(np.log1p(np.fft.fftshift(np.abs(np.fft.rfft2(self.buf)), axes=(0))+1e2))
        self.img_item.setImage(self.img[:,4:-5], autoLevels=True)

    def closeEvent(self, event):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = AudioSpectrogram()
    main.show()
    sys.exit(app.exec_())
