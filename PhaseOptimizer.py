import os
import cv2
import sys
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np


from scipy.misc import derivative
import matplotlib.pyplot as plt

from phase import generate_pishift
from camera import Camera
from optimize import centroid, get_warp, find_phase_subregion, avgpool, expand_array

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
        self._image = None
        self._beam = None
        self._graphic_figure = None
        self._mean_index = 0
        
        self.show_camera_line = True
        self.reset_mean = True
        self.capture_beam = True
        self.correction_value = None
        self.facecolor = '#FFFFFF'
        self.image_width = 640
        self.image_height = 480
        self.curve_width = 640
        self.curve_height = 480
        self.position_index = 0
        self.position_value = 1

    def set_phase(self, phase):
        self._camera.set_phase(phase)
        self.reset_mean = True

    def get_image(self):
        return self._image

    def get_image_beam(self):
        return self._beam

    def get_camera_shape(self):
        return self._camera.camera_shape

    def get_graphic_figure(self):
        return self._graphic_figure

    def run(self):
        x, y, yb = None, None, None
        gmin, gmax = np.inf, -np.inf
        while True:
            self._image = self._camera.get_image()
            if self._beam is None:
                self._beam = self._image
            if self.capture_beam:
                self._beam = self._mean_index/(self._mean_index + 1)*self._beam + self._image/(self._mean_index + 1)
                self._mean_index += 1

            h, w = self._image.shape

            figure = Figure()
            figure.patch.set_facecolor(self.facecolor)
            canvas = FigureCanvas(figure)
            axes = figure.gca()
            axes.imshow(self._image, cmap='gray')
            if self.show_camera_line:
                if self.position_index == 0:
                    axes.axvline(x = self.position_value - 1, linestyle='dotted', color='blue')
                else:
                    axes.axhline(y = self.position_value - 1, linestyle='dotted', color='blue')

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
                yb = self._beam[:, self.position_value - 1]
            else:
                y = self._image[self.position_value - 1, :]
                yb = self._beam[self.position_value - 1, :]
            axes.set_xlim(1, len(y) + 1)
            x = np.arange(1, len(y) + 1)
            ymin, ymax = np.min(self._image), np.max(self._image)
            if ymin < gmin: gmin = ymin
            if ymax > gmax: gmax = ymax
            if gmax > gmin:
                axes.set_ylim(gmin, gmax)
            axes.set_xlabel('Posicao')
            axes.set_ylabel('Intensidade')
            axes.plot(x, y, color='blue')
            axes.plot(x, yb, color='black')
            if self.correction_value is not None:
                axes.axhline(y = self.correction_value, color='blue', linestyle='dashed')

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

        self.factor = 1
        self.image_imin = 0
        self.image_imax = SLM_SHAPE[1]
        self.image_jmin = 0
        self.image_jmax = SLM_SHAPE[0]

        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.lblPhase.addAction(self.ui.actSavePhase)
        self.ui.lblPhase.addAction(self.ui.actUpdatePhase)
        self.ui.lblCamera.addAction(self.ui.actSaveImage)
        self.ui.lblCamera.addAction(self.ui.actShowCameraLine)
        self.ui.lblCameraSideView.addAction(self.ui.actSaveGraphic)
        self.ui.actLoadImage.triggered.connect(self.actLoadImageClicked)
        self.ui.actSavePhase.triggered.connect(self.actSavePhaseClicked)
        self.ui.actSaveImage.triggered.connect(self.actSaveImageClicked)
        self.ui.actShowCameraLine.triggered.connect(self.actShowCameraLineClicked)
        self.ui.actUpdatePhase.triggered.connect(self.actUpdatePhaseClicked)
        self.ui.actSaveGraphic.triggered.connect(self.actSaveGraphicClicked)
        self.ui.actAbout.triggered.connect(self.actAboutClicked)
        self.ui.cbPosition.currentIndexChanged.connect(self.cbPositionIndexChanged)
        self.ui.sbPositionValue.editingFinished.connect(self.sbPositionValueEditingFinished)
        self.ui.sbLineThickness.editingFinished.connect(self.sbPhaseEditingFinished)
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
        
        self.firstIter = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerTimeoutEvent)
        
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

    def timerTimeoutEvent(self):
        if self.firstIter:
            aligned_beam = cv2.warpAffine(self.initial_beam, self.warp, self.image.shape[::-1])
            self.image_goal = self.image.astype(np.float64)*self.ui.sbIntValue.value()/np.max(self.image)
            image_correction = self.image_goal.copy()
            
            image_sub = image_correction[self.image_imin:(self.image_imax + 1), self.image_jmin:(self.image_jmax + 1)]
            beam_sub = aligned_beam[self.image_imin:(self.image_imax + 1), self.image_jmin:(self.image_jmax + 1)]

            image_sub_pool = avgpool(image_sub, self.factor)
            self.__image_goal_sub_pool = image_sub_pool.copy()
            beam_sub_pool = avgpool(beam_sub, self.factor)
            
            image_sub_pool[np.nonzero(beam_sub_pool)] = image_sub_pool[np.nonzero(beam_sub_pool)]/beam_sub_pool[np.nonzero(beam_sub_pool)]
            image_sub_pool = 2*np.arcsin(np.clip(image_sub_pool, 0, 1))*128/np.pi

            image_sub_l = expand_array(image_sub_pool, self.factor)
            image_correction[self.image_imin:(self.image_imax + 1), self.image_jmin:(self.image_jmax + 1)] = image_sub_l

            self.image_correction = image_correction
            self.firstIter = False
        else:
            camera_image = cv2.warpAffine(self.camera.get_image(), self.warp, self.image.shape[::-1])
            
            camera_sub = camera_image[self.image_imin:(self.image_imax + 1), self.image_jmin:(self.image_jmax + 1)]
            image_sub = self.image_correction[self.image_imin:(self.image_imax + 1), self.image_jmin:(self.image_jmax + 1)]
            
            camera_sub_pool = avgpool(camera_sub, self.factor)
            image_sub_pool = avgpool(image_sub, self.factor)

            diff = camera_sub_pool - self.__image_goal_sub_pool
            image_sub_pool = image_sub_pool - np.sign(diff)
            image_sub_pool = np.clip(image_sub_pool, 0, 128)
            
            image_sub_l = expand_array(image_sub_pool, self.factor)
            self.image_correction[self.image_imin:(self.image_imax + 1), self.image_jmin:(self.image_jmax + 1)] = image_sub_l

        self.update(self.ui.sbLineThickness.value())

    def actLoadImageClicked(self):
        if self.initial_beam is None:
            self.initial_beam = self.camera.get_image_beam()
            self.camera.capture_beam = False
        filename = self.showOpenImageDialog()
        if filename:
            self.filename_image = filename
            self.loadImageFromFile(self.filename_image)

    def actSavePhaseClicked(self):
        filename = self.showSaveImageDialog()
        if filename:
            cv2.imwrite(filename, self.phase)

    def actShowCameraLineClicked(self, value):
        self.camera.show_camera_line = value

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
        self.update(self.ui.sbLineThickness.value())

    def pbCorrectClicked(self):
        if self.initial_beam is None:
            self.showDialog(QtWidgets.QMessageBox.Abort, 'Erro', 'Perfil do feixe não capturado.')
            return

        self.camera.correction_value = self.ui.sbIntValue.value()
        self.factor = self.ui.sbBlock.value()
        
        imin, imax, jmin, jmax = find_phase_subregion(self.image, self.factor)
        self.image_imin = imin
        self.image_imax = imax
        self.image_jmin = jmin
        self.image_jmax = jmax
        
        isActive = self.timer.isActive()
        self.ui.sbLineThickness.setEnabled(isActive)
        self.ui.pbAlign.setEnabled(isActive)
        self.ui.sbIntValue.setEnabled(isActive)
        self.ui.sbBlock.setEnabled(isActive)
        if not isActive:
            self.firstIter = True
            self.timer.start(2000)
            self.ui.pbCorrect.setText("Parar")
        else:
            self.timer.stop()
            self.ui.pbCorrect.setText("Corrigir")

    def pbAlignClicked(self):
        camera_image = self.camera.get_image()
        warp, match_res = get_warp(self.image, camera_image)
        self.warp = warp
        img = cv2.warpAffine(camera_image, self.warp, self.image.shape[::-1])
        cv2.imwrite('before.png', camera_image)
        cv2.imwrite('after.png', img)
        self.ui.lblConfidence.setText("Confidência: %.2f%%" % (100*match_res))
        self.ui.lblIntValue.setEnabled(True)
        self.ui.sbIntValue.setEnabled(True)
        self.ui.pbCorrect.setEnabled(True)

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
            h, w = np.array(image.shape)

            if np.any([h != SLM_SHAPE[0], w != SLM_SHAPE[1]]):
                self.showDialog(QtWidgets.QMessageBox.Warning, 'Aviso', 'Tamanho da imagem deve ser do tamanho da região do SLM (%d x %d)' % SLM_SHAPE)
                return

            self.image = image
            self.image_correction = image.copy()
            self.camera.correction_value = None

            self.update(self.ui.sbLineThickness.value())
            self.ui.gbPhase.setEnabled(True)

    def update(self, line_thickness):
        self.phase = generate_pishift(self.image_correction, line_thickness)
        self.camera.set_phase(self.phase)
        self.loadFigure(self.phase, self.ui.lblPhase)
        self.cbPositionIndexChanged()

app = QtWidgets.QApplication(sys.argv)

my_mainWindow = MainWindow()
my_mainWindow.show()

sys.exit(app.exec_())
