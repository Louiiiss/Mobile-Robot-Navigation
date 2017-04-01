from PyQt4 import QtGui  # (the example applies equally well to PySide)
import pyqtgraph as pg
import numpy as np
import Image
from pyqtgraph.Qt import QtCore, QtGui


testImage = Image.open("Megumin.png")
testImage.load()
imageData = np.asarray(testImage,dtype="int32")

#testImage = Image.open("Megumin.png")
app = QtGui.QApplication([])
# w = pg.GraphicsWindow(size=(1000,800), border=True)
# w.setWindowTitle('pyqtgraph example: ROI Examples')
#
# text = """Data Selection From Image.<br>\n
# Drag an ROI or its handles to update the selected image.<br>
# Hold CTRL while dragging to snap to pixel boundaries<br>
# and 15-degree rotation angles.
# """
# w1 = w.addLayout(row=0, col=0)
# label1 = w1.addLabel(text, row=0, col=0)
# v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
# img = pg.ImageItem(imageData)
# v1a.addItem(img)
# v1a.disableAutoRange('xy')
# v1a.autoRange()

pullData = open("sample.txt","r").read()
dataArray = pullData.split('\n')
xar = []
yar = []
for eachLine in dataArray:
    if len(eachLine)>1:
        x,y = eachLine.split(',')
        xar.append(int(x))
        yar.append(int(y))
plt = pg.plot(xar, yar, pen=None, symbol='o')
img = pg.ImageItem(imageData)
img.scale(0.1, 0.1)
img.setZValue(-100)
plt.addItem(img)

#pg.image(imageData)

app.exec_()
