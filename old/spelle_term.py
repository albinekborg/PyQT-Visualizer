
import sys
import numpy as np
import pyaudio

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QGraphicsScene, QGraphicsTextItem, QGraphicsView
from PyQt5.QtGui import QTextDocument, QTextOption, QColor
from PyQt5.QtCore import QTimer, QPointF, Qt
import pyqtgraph as pg

# Parameters for the audio stream
CHUNK = 200  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for microphone
RATE = 40000  # Samples per second


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
    
    def get_color(self,value, min_val, max_val):
        index = value/max_val - 0.021

        return self.colormap.map(index, mode='qcolor')
    
    def init_ui(self):
        self.setWindowTitle("Live Spectrogram")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        
        # Setup the spectrogram plot
        self.widget_1 = pg.PlotWidget()
        layout.addWidget(self.widget_1)
        self.img_item_1 = pg.ImageItem()
        self.widget_1.addItem(self.img_item_1)

        # Setup for Fourier Transform visualization
        self.widget_2 = pg.PlotWidget(title="Fourier Transform of Buffer")
        self.img_item_2 = pg.ImageItem()
        self.widget_2.addItem(self.img_item_2)
        layout.addWidget(self.widget_2)

        self.widget_1.getPlotItem().getAxis('left').setLogMode(False, True)
        self.widget_1.getPlotItem().vb.setLimits(yMin=1, yMax=CHUNK//2)  # You might need to adjust the upper limit based on your data


        # Setup an additional plot for random scatter points
        #self.widget_3 = pg.PlotWidget(title="Dynamic Line Plot")
        #layout.addWidget(self.widget_3)

        ### GPT Text  
        self.widget_3 = QGraphicsView()
        self.widget_3.setFixedHeight(150)
        self.widget_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Disable horizontal scrollbar
        self.widget_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Disable vertical scrollbar

        self.scene = QGraphicsScene()
        self.widget_3.setScene(self.scene)
        self.widget_3.setStyleSheet("background-color: black;")
        layout.addWidget(self.widget_3)
        self.text_items = []  # List to hold text items
        ## GPT TEXT END


        self.colormap = pg.colormap.get('nipy_spectral', source='matplotlib')
        lut = self.colormap.getLookupTable(0.0, 1.0, 256)
        self.img_item_1.setLookupTable(lut)
        self.img_item_2.setLookupTable(lut)

        container = QWidget()
        container.setStyleSheet("border: 1px solid #a0a0a0;background-color: black")
        container.setLayout(layout)
        self.setCentralWidget(container)

    
    def update(self):
        data = np.frombuffer(self.stream.read(CHUNK, exception_on_overflow=True), dtype=np.int16)
        self.tbuf = np.concatenate([data, self.tbuf[:-CHUNK]])
        self.yf = np.fft.rfft(self.tbuf * np.hanning(len(self.tbuf)))[1:]
        self.buf = np.concatenate([np.abs(self.yf[np.newaxis, :]), self.buf[:-1]], axis=0)
        
        buf_win = self.buf
        self.img = np.log1p(buf_win+1e3)
        self.img_item_1.setImage(self.img, autoLevels=True)
        
        win = np.exp(np.linspace(7,0,len(self.buf)))[:,np.newaxis]
        fft2 = np.abs(np.fft.fft2(self.buf*win))
        fft2_img = np.log1p(fft2+1e8)
        self.img_item_2.setImage(fft2_img, autoLevels=True)

        ## GPT TEXT

        bass = np.abs(np.mean(self.yf[:25]))
        mids = np.abs(np.mean(self.yf[25:75]))
        high = np.abs(np.mean(self.yf[75:100]))

        new_text = QGraphicsTextItem()
        text_option = QTextOption()
        text_option.setAlignment(Qt.AlignLeft)
        text_option.setTabStopDistance(150.0)  # Set tab stop distance to 150 pixels
        new_text.document().setDefaultTextOption(text_option)
        new_text.setTextWidth(self.widget_3.viewport().width())  # Ensure the text fills the width for proper alignment
        
        
        min_val = 0
        max_val = np.max([bass, mids, high])
       
        randsymbol = ['"','#','*','?','!','.',"@","+","-","0","x"]
        # Set text with HTML for color
        html_content = (
        f'<span style="font-family: \'Courier New\', monospace; color: {self.get_color(bass-70,min_val,max_val).name()};">{bass:.8f}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\t</span>'
        f'<span style="font-family: \'Courier New\', monospace; color: {self.get_color(mids,min_val,max_val).name()};">{mids:.8f}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\t</span>'
        f'<span style="font-family: \'Courier New\', monospace; color: {self.get_color(high+40,min_val,max_val).name()};"> {high:.8f}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\t</span>'
        f'<span style="font-family: \'Courier New\', monospace; color: #A0A0A0;"> {"".join([randsymbol[np.random.randint(len(randsymbol))] for _ in range(14)])}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\t</span>'
        )

        new_text.setHtml(html_content)

        #new_text.setHtml(
        #    f'<font color="{self.get_color(bass-20, min_val, max_val).name()}">{bass:.8f}</font><span style='
        #    '\t'
        #    f'<font color="{self.get_color(mids, min_val, max_val).name()}">{mids:.8f}</font>'
        #    f'<font color="{self.get_color(high, min_val, max_val).name()}">{high:.8f}</font>'
        #)


        self.scene.addItem(new_text)
        new_text.setPos(0, 200)  # Starting position at the bottom of the view
        self.text_items.append(new_text)

        # Move all text items up
        for item in self.text_items:
            pos = item.pos()
            item.setPos(pos.x(), pos.y() - 13) 
        self.text_items = [item for item in self.text_items if item.pos().y() > -self.widget_3.height()]
        ## GPT TEXT


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
