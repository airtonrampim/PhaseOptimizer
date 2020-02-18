import cv2
import sys
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np

from phase import generate_pishift
from camera import Camera
from optimize import centroid, align_image, get_warp

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtTest
import mainwindow

SLM_SHAPE = (1024, 1280)
BINARY = True

class CameraThread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(list)

    def __init__(self, shape):
        super().__init__()
        self._camera = Camera(shape)
        self._image = np.zeros(shape)
        
        self.image_width = 640
        self.image_height = 480
        self.curve_width = 640
        self.curve_height = 480
        self.position_index = 0
        self.position_value = 1

    def set_phase(self, phase):
        self._camera.set_phase(phase)
    
    def get_image(self):
        return self._image
    
    def get_camera_shape(self):
        return self._camera.camera_shape

    def run(self):
        x, y = None, None
        gmin, gmax = np.inf, -np.inf
        while True:
            self._image = self._camera.get_image()
            h, w = self._image.shape

            figure = Figure()
            canvas = FigureCanvas(figure)
            axes = figure.gca()
            axes.imshow(self._image, cmap='gray')
            canvas.draw()
            size = canvas.size()
            width, height = size.width(), size.height()
            convertToQtFormat = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)

            p1 = convertToQtFormat.scaled(self.image_width, self.image_height, QtCore.Qt.KeepAspectRatio)

            
            w = self.curve_width/matplotlib.rcParams["figure.dpi"]
            h = self.curve_height/matplotlib.rcParams["figure.dpi"]

            figure = Figure(figsize=(w, h))
            canvas = FigureCanvas(figure)
            axes = figure.gca()

            if self.position_index == 0:
                y = self._image[:, self.position_value - 1]
            else:
                y = self._image[self.position_value - 1, :]
            axes.set_xlim(1, len(y) + 1)
            x = np.arange(1, len(y) + 1)
            ymin, ymax = np.min(self._image), np.max(self._image)
            if ymin < gmin: gmin = ymin
            if ymax > gmax: gmax = ymax
            if gmax > gmin:
                axes.set_ylim(gmin, gmax)
            axes.set_xlabel('Posicao')
            axes.set_ylabel('Intensidade')
            axes.plot(x, y)

            canvas.draw()
            size = canvas.size()
            width, height = size.width(), size.height()
            p2 = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)

            self.changePixmap.emit([p1, p2])

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.phase = None
        self.image = None
        self.image_correction = None
        self.warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        self.x_ref = SLM_SHAPE[1]
        self.y_ref = SLM_SHAPE[0]

        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actLoadImage.triggered.connect(self.actLoadImageClicked)
        self.ui.cbPosition.currentIndexChanged.connect(self.cbPositionIndexChanged)
        self.ui.sbPositionValue.editingFinished.connect(self.sbPositionValueEditingFinished)
        self.ui.pbUpdate.clicked.connect(self.pbUpdateClicked)
        self.ui.sbX.editingFinished.connect(self.pbUpdateClicked)
        self.ui.sbY.editingFinished.connect(self.pbUpdateClicked)
        self.ui.pbOptimize.clicked.connect(self.pbOptimizeClicked)
        self.ui.pbAlign.clicked.connect(self.pbAlignClicked)
        
        self.camera = CameraThread(SLM_SHAPE)
        self.camera.changePixmap.connect(self.setImage)
        self.camera.image_height = self.ui.lblCamera.height()
        self.camera.image_width = self.ui.lblCamera.width()
        self.camera.curve_height = self.ui.lblCameraSideView.height()
        self.camera.curve_width = self.ui.lblCameraSideView.width()
        self.camera.start()
        
        self.cbPositionIndexChanged()

        self.ui.sbPositionValue.setValue(self.camera.get_camera_shape()[1 - self.ui.cbPosition.currentIndex()]//2)

    @QtCore.pyqtSlot(list)
    def setImage(self, images):
        self.ui.lblCamera.setPixmap(QtGui.QPixmap.fromImage(images[0]))
        self.ui.lblCameraSideView.setPixmap(QtGui.QPixmap.fromImage(images[1]))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.camera.image_height = self.ui.lblCamera.height()
        self.camera.image_width = self.ui.lblCamera.width()
        self.camera.curve_height = self.ui.lblCameraSideView.height()
        self.camera.curve_width = self.ui.lblCameraSideView.width()

    def showImageFileDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.HideNameFilterDetails
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Abrir imagem", "", "Imagem (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.webp *.pbm *.pgm *.ppm *.pxm *.pnm *.pfm *.sr *.ras *.tiff *.tif *.exr *.hdr *.pic);;Todos os arquivos (*)", options = options)        
        return filename

    def loadFigure(self, array, label):
        w = label.width()/matplotlib.rcParams["figure.dpi"]
        h = label.height()/matplotlib.rcParams["figure.dpi"]

        figure = Figure(figsize=(w, h))
        canvas = FigureCanvas(figure)
        axes = figure.gca()
        axes.imshow(array, cmap='gray')
        canvas.draw()
        size = canvas.size()
        width, height = size.width(), size.height()
        image = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)
        label.setPixmap(QtGui.QPixmap.fromImage(image))


    def showDialog(self, icon, title, message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(icon)
        msgBox.setText(message)
        msgBox.setWindowTitle(title)
        msgBox.exec()

    def updateWarpPosition(self):
        self.warp[0,-1] = 302 - self.ui.sbX.value()
        self.warp[1,-1] = 308 - self.ui.sbY.value()

    def cbPositionIndexChanged(self):
        self.ui.sbPositionValue.setMaximum(self.camera.get_camera_shape()[1 - self.ui.cbPosition.currentIndex()])
        self.camera.position_index = self.ui.cbPosition.currentIndex()
        self.camera.position_value = self.ui.sbPositionValue.value()

    def sbPositionValueEditingFinished(self):
        self.camera.position_index = self.ui.cbPosition.currentIndex()
        self.camera.position_value = self.ui.sbPositionValue.value()

    def actLoadImageClicked(self):
        filename = self.showImageFileDialog()
        if filename:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.image = image
            self.image_correction = image

            camera_image = self.camera.get_image()
            x_c, y_c = centroid(camera_image/255.)
            x_i, y_i = centroid(image/255.)

            h, w = np.array(image.shape)

            if np.any([h > SLM_SHAPE[0], w > SLM_SHAPE[1]]):
                self.showDialog(QtWidgets.QMessageBox.Warning, 'Aviso', 'Tamanho da imagem deve ser menor que a região do SLM (%d x %d)' % SLM_SHAPE)
                return
            
            y_max, x_max = SLM_SHAPE

            self.ui.sbX.setMaximum(x_max - w)
            self.ui.sbX.setValue(int((x_c*SLM_SHAPE[1]/self.camera.get_camera_shape()[1] - x_i)))
            self.ui.sbY.setMaximum(y_max - h)
            self.ui.sbY.setValue(int((y_c*SLM_SHAPE[0]/self.camera.get_camera_shape()[0] - y_i)))

            opt_args = (self.ui.sbX.value(), self.ui.sbY.value(), None, None)
            phase = generate_pishift(image, opt_args = opt_args, slm_shape = SLM_SHAPE, binary = BINARY)
            self.phase = phase
            self.camera.set_phase(phase)
            camera_image = self.camera.get_image()
    
            self.loadFigure(self.phase, self.ui.lblPhase)
            self.cbPositionIndexChanged()

            self.ui.gbPhase.setEnabled(True)
            self.ui.gbCameraSideView.setEnabled(True)

    def update(self, opt_args):
        self.phase = generate_pishift(self.image_correction, opt_args = opt_args, slm_shape = SLM_SHAPE, binary = BINARY)
        self.camera.set_phase(self.phase)
        camera_image = self.camera.get_image()

        self.loadFigure(self.phase, self.ui.lblPhase)
        self.cbPositionIndexChanged()

    def pbUpdateClicked(self):
        opt_args = (self.ui.sbX.value(), self.ui.sbY.value(), None, None)
        self.update(opt_args)

    def pbOptimizeClicked(self):
        camera_image = self.camera.get_image()
        self.updateWarpPosition()
        aligned_image = align_image(self.image, camera_image, self.warp)
        self.image_correction = self.image - aligned_image
        
        opt_args = self.ui.sbX.value(), self.ui.sbY.value(), None, None
        self.update(opt_args)
    
    def pbAlignClicked(self):
        camera_image = self.camera.get_image()
        warp, match_res = get_warp(self.image, camera_image)
        self.x_ref = self.ui.sbX.value()
        self.y_ref = self.ui.sbY.value()
        self.warp = warp
        self.ui.lblConfidence.setText("Nível de confidência: %.2f%%" % (100*match_res))

app = QtWidgets.QApplication(sys.argv)

my_mainWindow = MainWindow()
my_mainWindow.show()

sys.exit(app.exec_())
