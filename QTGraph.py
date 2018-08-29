from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from scipy.signal import butter, lfilter, freqz, argrelextrema
import numpy as np
import threading
import time
import sys

streams = list()

# numpy parameters
order = 2
fs = 100.0
cutoff = 1.8
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

class myQWidget(QtGui.QWidget):
    def __init__(self):
        super(myQWidget, self).__init__()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Enter or e.key() == QtCore.Qt.Key_Space:
            self.event.set()

    def setKeyPressTarget(self, thread, evt):
        self.kpThread = thread
        self.event = evt
        self.kpThread.start()

class QTGraph():
    def __init__(self, sources, sourcesName, sourcesNum=1, title='curve graph',\
                    width=600, height=600, x_range=500, y_range=1, y_central=0.5, lock=None, lowpass=False):
        
        # set data sources
        self.title = title
        self.sources = sources
        self.sourcesName = sourcesName
        self.sourcesNum = sourcesNum
        self.lock = lock

        # graph configure
        self.width = width
        self.height = height
        self.x_range = x_range
        self.y_range = y_range
        self.y_central = y_central
        self.x_coordinate = [i for i in range(self.x_range)]
        self.app = QtGui.QApplication(sys.argv)
        self.mainWindow = QtGui.QMainWindow()
        self.mainWindow.setWindowTitle(self.title)
        self.mainWindow.resize(self.width, self.height)
        self.mainWidget = myQWidget()
        self.mainWindow.setCentralWidget(self.mainWidget)

        self.layout = QtGui.QVBoxLayout()
        self.mainWidget.setLayout(self.layout)

        # curve in plot widgets configure
        self.showLowPass = lowpass
        self.curves = []
        for i in range(self.sourcesNum):
            plotWidget = pg.PlotWidget(name=self.sourcesName[i])
            plotWidget.setXRange(0, self.x_range)
            plotWidget.setYRange(self.y_central-float(self.y_range)/2, self.y_central+float(self.y_range)/2)
            self.layout.addWidget(plotWidget)
            
            curve = plotWidget.plot()
            self.curves.append(curve)

    def update(self):
        if self.lock != None:
            self.lock.acquire()
        for src, curve in zip(self.sources, self.curves):
            if len(src) < self.x_range: 
                fill = [np.NAN for i in range(self.x_range-len(src))]
                if self.showLowPass == False: 
                    curve.setData(y=fill+src, x=self.x_coordinate)
                else:
                    disp = butter_lowpass_filter(src, cutoff, fs, order)
                    curve.setData(y=fill+disp.tolist(), x=self.x_coordinate)
            else: 
                if self.showLowPass == False: 
                    curve.setData(y=src, x=self.x_coordinate)
                else:
                    disp = butter_lowpass_filter(src, cutoff, fs, order)
                    curve.setData(y=disp.tolist(), x=self.x_coordinate)
        if self.lock != None:
            self.lock.release()
        # self.app.processEvents()

    def show(self, animate=False):
        self.mainWindow.show()
        if animate:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(50)
            self.app.instance().exec_()
        else:
            self.update()
            self.app.instance().exec_()

    def close(self):
        self.app.closeAllWindows()
        self.app.quit()

    def setTask(self, thread, evt):
        self.mainWidget.setKeyPressTarget(thread, evt)

if __name__ == '__main__':

    # files = ['ax.txt', 'ay.txt', 'az.txt']
    # fps = [open(file, 'r') for file in files]

    # for file, fp in zip(files,fps):
    #     print(file)
    #     stream = list()
    #     for line in fp.readlines():
    #         stream += [float(x) for x in line.split(' ')]
    #     fp.close()
    #     streams.append(stream)
    import argparse
    import json
    import math

    parser = argparse.ArgumentParser(description='process input parameters')
    parser.add_argument('fn', help='input the filename you want to show', type=str)
    args = parser.parse_args()

    filename = args.fn
    fp = open(filename, 'r')
    word = filename.split('.')[0].split('/')[-1]
    jsdata = json.loads(fp.read())
    fp.close()

    valid_curve_length = min( [ len(c) for c in [ jsdata["gx"], jsdata["gy"], jsdata["gz"] ] ] )
    resultant_gyro = [ math.sqrt( jsdata["gx"][i]**2 + jsdata["gy"][i]**2 + jsdata["gz"][i]**2 ) for i in range( valid_curve_length ) ]
    streams.append(resultant_gyro)

    qtGraph = QTGraph(streams, [filename], 1, x_range=max([len(l) for l in streams]))
    qtGraph.show(animate=False)
    qtGraph.close()

