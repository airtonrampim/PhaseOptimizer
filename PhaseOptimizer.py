import os
import cv2
import sys
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np

import matplotlib.pyplot as plt

from phase import generate_pishift
from camera import Camera
from optimize import centroid, align_image, get_warp

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtTest
import mainwindow
import AboutWindow

SLM_SHAPE = (1024, 1280)

class CameraThread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(list)

    def __init__(self, shape):
        super().__init__()
        self._camera = Camera(shape)
        self._image = np.zeros(shape)
        self._graphic_figure = None
        
        self.facecolor = '#FFFFFF'
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
    
    def get_graphic_figure(self):
        return self._graphic_figure

    def run(self):
        x, y = None, None
        gmin, gmax = np.inf, -np.inf
        while True:
            self._image = self._camera.get_image()
            h, w = self._image.shape

            figure = Figure()
            figure.patch.set_facecolor(self.facecolor)
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

            self._graphic_figure = Figure(figsize=(w, h))
            self._graphic_figure.patch.set_facecolor(self.facecolor)
            canvas = FigureCanvas(self._graphic_figure)
            axes = self._graphic_figure.gca()

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
    
    def close(self):
        self._camera.close()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.about = AboutWindow.AboutWindow(self)
        
        self.phase = None
        self.image = None
        self.image_correction = None
        self.filename_image = None
        self.warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        self.x_ref = SLM_SHAPE[1]
        self.y_ref = SLM_SHAPE[0]

        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.lblPhase.addAction(self.ui.actSavePhase)
        self.ui.lblPhase.addAction(self.ui.actUpdatePhase)
        self.ui.lblCamera.addAction(self.ui.actSaveImage)
        self.ui.lblCameraSideView.addAction(self.ui.actSaveGraphic)
        self.ui.actLoadImage.triggered.connect(self.actLoadImageClicked)
        self.ui.actSavePhase.triggered.connect(self.actSavePhaseClicked)
        self.ui.actSaveImage.triggered.connect(self.actSaveImageClicked)
        self.ui.actUpdatePhase.triggered.connect(self.actUpdatePhaseClicked)
        self.ui.actSaveGraphic.triggered.connect(self.actSaveGraphicClicked)
        self.ui.actAbout.triggered.connect(self.actAboutClicked)
        self.ui.cbPosition.currentIndexChanged.connect(self.cbPositionIndexChanged)
        self.ui.sbPositionValue.editingFinished.connect(self.sbPositionValueEditingFinished)
        self.ui.sbX.editingFinished.connect(self.sbCoordsGratingEditingFinished)
        self.ui.sbY.editingFinished.connect(self.sbCoordsGratingEditingFinished)
        self.ui.sbDistanceGrating.editingFinished.connect(self.sbCoordsGratingEditingFinished)
        self.ui.sbLengthGrating.editingFinished.connect(self.sbCoordsGratingEditingFinished)
        self.ui.cbBinary.clicked.connect(self.sbCoordsGratingEditingFinished)
        self.ui.pbOptimize.clicked.connect(self.pbOptimizeClicked)
        self.ui.pbAlign.clicked.connect(self.pbAlignClicked)
        
        self.camera = CameraThread(SLM_SHAPE)
        self.camera.changePixmap.connect(self.setImage)
        self.camera.facecolor = self.palette().window().color().name()
        self.camera.image_height = self.ui.lblCamera.height()
        self.camera.image_width = self.ui.lblCamera.width()
        self.camera.curve_height = self.ui.lblCameraSideView.height()
        self.camera.curve_width = self.ui.lblCameraSideView.width()
        self.camera.start()
        
        self.cbPositionIndexChanged()

        self.ui.sbPositionValue.setValue(self.camera.get_camera_shape()[1 - self.ui.cbPosition.currentIndex()]//2)
        self.ui.sbDistanceGrating.setMaximum(SLM_SHAPE[0])
        self.ui.sbLengthGrating.setMaximum(SLM_SHAPE[0])
        
    # Multithread events ----------------------------------------------------------------------------------------------

    @QtCore.pyqtSlot(list)
    def setImage(self, images):
        self.ui.lblCamera.setPixmap(QtGui.QPixmap.fromImage(images[0]))
        self.ui.lblCameraSideView.setPixmap(QtGui.QPixmap.fromImage(images[1]))

    # Application events ----------------------------------------------------------------------------------------------

    def closeEvent(self, event):
        super().closeEvent(event)
        self.camera.close()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.camera.image_height = self.ui.lblCamera.height()
        self.camera.image_width = self.ui.lblCamera.width()
        self.camera.curve_height = self.ui.lblCameraSideView.height()
        self.camera.curve_width = self.ui.lblCameraSideView.width()

    def actLoadImageClicked(self):
        filename = self.showOpenImageDialog()
        if filename:
            self.filename_image = filename
            self.loadImageFromFile(self.filename_image)

    def actSavePhaseClicked(self):
        filename = self.showSaveImageDialog()
        if filename:
            cv2.imwrite(filename, self.phase)

    def actSaveImageClicked(self):
        camera_image = self.camera.get_image()
        filename = self.showSaveImageDialog()
        if filename:
            cv2.imwrite(filename, camera_image)

    def actUpdatePhaseClicked(self):
        if os.path.isfile(self.filename_image):
            self.loadImageFromFile(self.filename_image)
        else:
            self.showDialog(QtWidgets.QMessageBox.Warning, 'Aviso', 'Arquivo %s não encontrado' % self.filename_image)

    def actSaveGraphicClicked(self):
        figure = self.camera.get_graphic_figure()
        filename = self.showSaveImageDialog()
        if filename:
            figure.savefig(filename)

    def actAboutClicked(self):
        self.about.show()

    def cbPositionIndexChanged(self):
        self.ui.sbPositionValue.setMaximum(self.camera.get_camera_shape()[1 - self.ui.cbPosition.currentIndex()])
        self.camera.position_index = self.ui.cbPosition.currentIndex()
        self.camera.position_value = self.ui.sbPositionValue.value()

    def sbPositionValueEditingFinished(self):
        self.camera.position_index = self.ui.cbPosition.currentIndex()
        self.camera.position_value = self.ui.sbPositionValue.value()

    def sbCoordsGratingEditingFinished(self):
        coords = self.ui.sbX.value(), self.ui.sbY.value()
        grating_params = self.ui.sbDistanceGrating.value(), self.ui.sbLengthGrating.value()
        self.update(coords, grating_params)

    def pbOptimizeClicked(self):
        camera_image = self.camera.get_image()
        self.updateWarpPosition()

        aligned_image = align_image(self.image, camera_image, self.warp)
        img_min, img_max = np.min(self.image), np.max(self.image)
        cam_min, cam_max = np.min(aligned_image), np.max(aligned_image)
        self.ui.lblConfidence.setText("Contraste: %d - %d (imagem)/%d - %d (câmera)" % (img_min, img_max, cam_min, cam_max))


        factor = self.ui.dsbContrast.value()/100.
        self.image_correction =  (self.image - factor*aligned_image).astype(self.image.dtype)

        coords = self.ui.sbX.value(), self.ui.sbY.value()
        grating_params = self.ui.sbDistanceGrating.value(), self.ui.sbLengthGrating.value()
        self.update(coords, grating_params)

    def pbAlignClicked(self):
        camera_image = self.camera.get_image()
        warp, match_res = get_warp(self.image, camera_image)
        self.x_ref = self.ui.sbX.value()
        self.y_ref = self.ui.sbY.value()
        self.warp = warp
        img_min, img_max = np.min(self.image), np.max(self.image)
        cam_min, cam_max = np.min(camera_image), np.max(camera_image)
        self.ui.lblConfidence.setText("Confidência: %.2f%%. Contraste: %d - %d (imagem)/%d - %d (câmera)" % (100*match_res, img_min, img_max, cam_min, cam_max))

    # Other procedures -------------------------------------------------------------------------------------------------

    def showSaveImageDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.HideNameFilterDetails
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption="Salvar imagem", filter="Imagem (*)", options = options)
        return filename

    def showOpenImageDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.HideNameFilterDetails
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="Abrir imagem", filter="Imagem (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.webp *.pbm *.pgm *.ppm *.pxm *.pnm *.pfm *.sr *.ras *.tiff *.tif *.exr *.hdr *.pic);;Todos os arquivos (*)", options = options)
        return filename

    def loadFigure(self, array, label):
        w = label.width()/matplotlib.rcParams["figure.dpi"]
        h = label.height()/matplotlib.rcParams["figure.dpi"]

        figure = Figure(figsize=(w, h))
        canvas = FigureCanvas(figure)
        axes = figure.gca()

        facecolor = self.palette().window().color()
        figure.patch.set_facecolor(facecolor.name())
        axes.imshow(array, cmap='gray')
        canvas.draw()
        size = canvas.size()
        width, height = size.width(), size.height()
        image = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)
        label.setPixmap(QtGui.QPixmap.fromImage(image))

    def showDialog(self, icon, title, message):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(icon)
        msgBox.setText(message)
        msgBox.setWindowTitle(title)
        msgBox.exec()

    def updateWarpPosition(self):
        pass
        #TODO: estudar mais sobre transformada afim para consertar a translacao

        #self.warp[0,-1] = 302 - self.ui.sbX.value() #290
        #self.warp[1,-1] = 308 - self.ui.sbY.value() #296

    def loadImageFromFile(self, filename):
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.image = image
            self.image_correction = image

            camera_image = self.camera.get_image()

            if np.abs(camera_image).sum() > 0: x_c, y_c = centroid(camera_image/255.)
            else: x_c, y_c = camera_image.shape[1]/2, camera_image.shape[0]/2
            if np.abs(image).sum() > 0: x_i, y_i = centroid(image/255.)
            else: x_i, y_i = image.shape[1]/2, image.shape[0]/2

            h, w = np.array(image.shape)

            if np.any([h > SLM_SHAPE[0], w > SLM_SHAPE[1]]):
                self.showDialog(QtWidgets.QMessageBox.Warning, 'Aviso', 'Tamanho da imagem deve ser menor que a região do SLM (%d x %d)' % SLM_SHAPE)
                return
            
            y_max, x_max = SLM_SHAPE

            self.ui.sbX.setMaximum(x_max - w)
            self.ui.sbX.setValue(int((x_c*SLM_SHAPE[1]/self.camera.get_camera_shape()[1] - x_i)))
            self.ui.sbY.setMaximum(y_max - h)
            self.ui.sbY.setValue(int((y_c*SLM_SHAPE[0]/self.camera.get_camera_shape()[0] - y_i)))

            coords = self.ui.sbX.value(), self.ui.sbY.value()
            grating_params = self.ui.sbDistanceGrating.value(), self.ui.sbLengthGrating.value()
            phase = generate_pishift(image, coords = coords, shape = SLM_SHAPE, binary = self.ui.cbBinary.isChecked(), grating_params = grating_params)
            self.phase = phase
            self.camera.set_phase(phase)
            camera_image = self.camera.get_image()
    
            self.loadFigure(self.phase, self.ui.lblPhase)
            self.cbPositionIndexChanged()
            
            self.ui.gbPhase.setEnabled(True)

    def update(self, coords, grating_params):
        self.phase = generate_pishift(self.image_correction, coords = coords, shape = SLM_SHAPE, binary = self.ui.cbBinary.isChecked(), grating_params = grating_params)
        self.camera.set_phase(self.phase)
        camera_image = self.camera.get_image()

        self.loadFigure(self.phase, self.ui.lblPhase)
        self.cbPositionIndexChanged()

app = QtWidgets.QApplication(sys.argv)

my_mainWindow = MainWindow()
my_mainWindow.show()

sys.exit(app.exec_())
