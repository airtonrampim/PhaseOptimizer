import os
import cv2
import sys
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np

#TODO: Remover
#from matplotlib.patches import ConnectionPatch

from scipy.misc import derivative
import matplotlib.pyplot as plt

from phase import generate_pishift, adjust_image
from camera import Camera
from optimize import centroid, align_image, get_warp

from PyQt5.QtCore import qDebug

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtTest
import mainwindow
import AboutWindow

SLM_SHAPE = (1024, 1272)

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
            
            #TODO: Remover
            #axes.add_patch(plt.Circle((600, 550), radius=50, edgecolor='b', facecolor='none'))
            
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

            # TODO: Remover
            #axes.add_patch(ConnectionPatch((600, ymin), (600, ymax), "data", "data", color='b'))

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
        
        self.initial_beam = None
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
        self.ui.sbLineThickness.editingFinished.connect(self.sbPhaseEditingFinished)
        self.ui.dsbPhaseValue.editingFinished.connect(self.sbPhaseEditingFinished)
        self.ui.pbCorrect.clicked.connect(self.pbCorrectClicked)
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
        if self.initial_beam is None:
            self.initial_beam = self.camera.get_image()
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

    def sbPhaseEditingFinished(self):
        self.update(self.ui.sbLineThickness.value(), self.ui.dsbPhaseValue.value())

    def pbCorrectClicked(self):
        camera_image = self.camera.get_image()

        if self.initial_beam is None:
            self.showDialog(QtWidgets.QMessageBox.Abort, 'Erro', 'Perfil de intensidade não capturado.')
            return

        mask = np.where(self.image_correction > 0, 1, 0) # Remove o fundo da imagem
        kernel = np.ones((20,20), dtype=np.uint8)
        image_l = cv2.erode(self.image_correction, kernel, iterations=1)
        mask_l = np.where(image_l > 0, 1, 0) # Remove os cantos da imagem

        #aligned_camera_image = mask*align_image(self.image_correction, camera_image, self.warp)

        #img_min, img_max = np.min(self.image_correction), np.max(self.image_correction)
        #cam_min, cam_max = np.min(aligned_camera_image), np.max(aligned_camera_image)
        #self.ui.lblConfidence.setText("Contraste: %d - %d (imagem)/%d - %d (câmera)" % (img_min, img_max, cam_min, cam_max))
        
        aligned_beam = align_image(self.image_correction, self.initial_beam, self.warp)

        self.image_correction = np.divide(self.image, aligned_beam, out=np.zeros_like(self.image), where=aligned_beam!=0)

        self.update(self.ui.sbLineThickness.value(), self.ui.dsbPhaseValue.value())

    def pbAlignClicked(self):
        camera_image = self.camera.get_image()
        warp, match_res = get_warp(self.image, camera_image)
        self.warp = warp
        self.ui.lblConfidence.setText("Confidência: %.2f%%" % (100*match_res))

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

    def loadImageFromFile(self, filename):
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            camera_image = self.camera.get_image()

            if np.abs(camera_image).sum() > 0: x_c, y_c = centroid(camera_image/255.)
            else: x_c, y_c = camera_image.shape[1]/2, camera_image.shape[0]/2
            if np.abs(image).sum() > 0: x_i, y_i = centroid(image/255.)
            else: x_i, y_i = image.shape[1]/2, image.shape[0]/2

            h, w = np.array(image.shape)

            if np.any([h > SLM_SHAPE[0], w > SLM_SHAPE[1]]):
                self.showDialog(QtWidgets.QMessageBox.Warning, 'Aviso', 'Tamanho da imagem deve ser menor que a região do SLM (%d x %d)' % SLM_SHAPE)
                return

            coords = int((x_c*SLM_SHAPE[1]/self.camera.get_camera_shape()[1] - x_i)), int((y_c*SLM_SHAPE[0]/self.camera.get_camera_shape()[0] - y_i))
            image_slm = adjust_image(image, coords = coords, shape = SLM_SHAPE)
            self.image = image_slm
            self.image_correction = image_slm.copy()

            self.update(self.ui.sbLineThickness.value(), self.ui.dsbPhaseValue.value())            
            self.ui.gbPhase.setEnabled(True)

    def update(self, line_thickness, phase_max):
        self.phase = generate_pishift(self.image_correction, line_thickness, phase_max)
        self.camera.set_phase(self.phase)
        self.loadFigure(self.phase, self.ui.lblPhase)
        self.cbPositionIndexChanged()

app = QtWidgets.QApplication(sys.argv)

my_mainWindow = MainWindow()
my_mainWindow.show()

sys.exit(app.exec_())
