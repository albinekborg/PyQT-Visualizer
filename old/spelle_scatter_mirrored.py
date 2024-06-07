import sys
import numpy as np
import pyaudio
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# Parameters for the audio stream
CHUNK = 200  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for microphone
RATE = 10000  # Samples per second

class AudioSpectrogram(QMainWindow):
    def __init__(self):
        super().__init__()
        self.xx = None
        self.x = []
        self.y = []
        self.x1 = []
        self.y1 = []
        self.x2 = []
        self.y2 = []
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
        self.buf_phase = np.zeros((self.buf_size, CHUNK//2))  # Ensure the buffer is properly shaped
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

        # Setup for Fourier Transform visualization
        self.empty_plot = pg.PlotWidget(title="Fourier Transform of Buffer")
        self.empty_img_item = pg.ImageItem()
        self.empty_plot.addItem(self.empty_img_item)
        layout.addWidget(self.empty_plot)

        self.plot_widget.getPlotItem().getAxis('left').setLogMode(False, True)
        self.plot_widget.getPlotItem().vb.setLimits(yMin=1, yMax=CHUNK//2)  # You might need to adjust the upper limit based on your data


        # Setup an additional plot for random scatter points
        self.line_plot_widget = pg.PlotWidget(title="Dynamic Line Plot")
        self.line_plot_item = self.line_plot_widget.plot(pen=pg.mkPen(color=(100, 100, 250), width=2))
        self.line_plot_item1 = self.line_plot_widget.plot(pen=pg.mkPen(color=(100, 100, 250), width=2))
        self.line_plot_item2 = self.line_plot_widget.plot(pen=pg.mkPen(color=(100, 100, 250), width=2))
        # self.line_plot_item.setLimits(yMin=-0.01, yMax=0.01, xMin=-0.01, xMax=0.01)  # You might need to adjust the upper limit based on your data
        layout.addWidget(self.line_plot_widget)


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
        self.yf = np.fft.rfft(self.tbuf * np.hanning(len(self.tbuf)))[1:]
        self.buf = np.concatenate([np.abs(self.yf[np.newaxis, :]), self.buf[:-1]], axis=0)
        
        buf_win = self.buf
        self.img = np.log1p(buf_win+1e3)
        self.img_item.setImage(self.img, autoLevels=True)
        
        win = np.exp(np.linspace(7,0,len(self.buf)))[:,np.newaxis]
        fft2 = np.abs(np.fft.fft2(self.buf*win))
        fft2_img = np.log1p(fft2+1e8)
        self.empty_img_item.setImage(fft2_img, autoLevels=True)

        if self.xx is None:
            self.xx = np.random.normal(size=(len(np.ndarray.flatten(fft2_img))))
            self.yy = np.random.normal(size=(len(np.ndarray.flatten(fft2_img))))
            self.xx1 = np.random.normal(size=(len(np.ndarray.flatten(fft2_img))))
            self.yy1 = np.random.normal(size=(len(np.ndarray.flatten(fft2_img))))
            self.xx2 = np.random.normal(size=(len(np.ndarray.flatten(fft2_img))))
            self.yy2 = np.random.normal(size=(len(np.ndarray.flatten(fft2_img))))

        self.x.append( self.xx[np.newaxis,:]   @ np.exp(np.ndarray.flatten(fft2_img)[:,np.newaxis,]) / np.sum(fft2_img))
        self.y.append( self.yy[np.newaxis,:]   @ np.exp(np.ndarray.flatten(fft2_img)[:,np.newaxis,]) / np.sum(fft2_img))
        self.x1.append( self.xx1[np.newaxis,:] @ np.exp(np.ndarray.flatten(fft2_img)[:,np.newaxis,]) / np.sum(fft2_img))
        self.y1.append( self.yy1[np.newaxis,:] @ np.exp(np.ndarray.flatten(fft2_img)[:,np.newaxis,]) / np.sum(fft2_img))
        self.x2.append( self.xx2[np.newaxis,:] @ np.exp(np.ndarray.flatten(fft2_img)[:,np.newaxis,]) / np.sum(fft2_img))
        self.y2.append( self.yy2[np.newaxis,:] @ np.exp(np.ndarray.flatten(fft2_img)[:,np.newaxis,]) / np.sum(fft2_img))
        
        # self.scatter_plot_item.clear()
        self.line_plot_item.setData(np.array(self.x).flatten(), np.array(self.y).flatten())
        self.line_plot_item1.setData(np.array(self.x1).flatten(), np.array(self.y1).flatten())
        self.line_plot_item2.setData(np.array(self.x2).flatten(), np.array(self.y2).flatten())
        if len(self.x) > 10:
            self.x.pop(0)
            self.y.pop(0)
            self.x1.pop(0)
            self.y1.pop(0)
            self.x2.pop(0)
            self.y2.pop(0)

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
