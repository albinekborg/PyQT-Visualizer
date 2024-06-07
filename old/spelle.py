import sys
import numpy as np
import pyaudio
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# Parameters for the audio stream
CHUNK = 128  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for microphone
RATE = 44100 // 8  # Samples per second

class AudioSpectrogram(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_audio()
        self.start_timer()

    def setup_audio(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.prev_img = None
        self.yf = 0
        self.buf_size = 256
        self.buf = np.zeros((self.buf_size, CHUNK//2))  # Ensure the buffer is properly shaped
        self.buf_iter = 0
        self.tbuf = np.zeros(CHUNK)

    def start_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(0)  # Milliseconds interval for timer
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def init_ui(self):
        self.setWindowTitle("Live Spectrogram")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        
        # Setup the spectrogram plot
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)

        self.empty_plot = pg.PlotWidget(title="Fourier Transform of Buffer")
        self.empty_img_item = pg.ImageItem()
        self.empty_plot.addItem(self.empty_img_item)
        layout.addWidget(self.empty_plot)
        
        # Setup an additional empty plot
        # self.empty_plot = pg.PlotWidget(title="abece")
        # layout.addWidget(self.empty_plot)

        colormap = pg.colormap.get('nipy_spectral', source='matplotlib')
        lut = colormap.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(lut)
        self.empty_img_item.setLookupTable(lut)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update(self):
        data = np.frombuffer(self.stream.read(CHUNK, exception_on_overflow=True), dtype=np.int16)
        self.tbuf = np.concatenate([data, self.tbuf[:-CHUNK]])
        self.yf = np.abs(np.fft.rfft(self.tbuf * np.hanning(len(self.tbuf))))[1:]
        self.buf = np.concatenate([self.yf[np.newaxis, :], self.buf[:-1]], axis=0)
        
        buf_win = self.buf
        self.img = np.log1p(buf_win+1e3)
        self.img_item.setImage(self.img, autoLevels=True)
        win = np.exp(np.linspace(3,0,len(self.buf)))[:,np.newaxis]
        fft2 = np.abs(np.fft.fft2(self.buf*win))
        fft2_img = np.log1p(fft2+1e9)
        self.empty_img_item.setImage(fft2_img, autoLevels=True)

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
