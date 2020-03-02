#https://stackoverflow.com/a/57613878
#https://stackoverflow.com/a/2308718
import cv2
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui

class SLMdisplay(QtWidgets.QFrame):
    def __init__(self, image_shape, icon = None, monitor = 1):
        super().__init__()
        if icon is not None:
            self.setWindowIcon(icon)
        self.image_shape = image_shape
        self._label = QtWidgets.QLabel(self)
        self._label.setText("")
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setObjectName("_label")
        self._layout = QtWidgets.QGridLayout(self)
        self._layout.addWidget(self._label)
        monitor_geometry = QtWidgets.QDesktopWidget().screenGeometry(monitor)
        self.move(monitor_geometry.left(), monitor_geometry.top())
        self.showFullScreen()

    def updateArray(self, array):
        image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        height, width, ch = image.shape
        image_height, image_width = self.image_shape
        image = QtGui.QImage(image.data, width, height, width*ch, QtGui.QImage.Format_RGB888)
        image_scaled = image.scaled(image_width, image_height, QtCore.Qt.KeepAspectRatio)
        self._label.setPixmap(QtGui.QPixmap.fromImage(image_scaled))
