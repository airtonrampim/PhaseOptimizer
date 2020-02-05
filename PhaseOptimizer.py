import cv2
import sys
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np

from phase import generate_pishift
from camera import Camera
from optimize import centroid, align_image

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtTest
import mainwindow

SLM_SHAPE = (1024, 1280)
OVERWRITE = False
BINARY = True

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.phase = None
        self.camera = Camera(SLM_SHAPE)
        self.image = None
        self.image_correction = None

        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actLoadImage.triggered.connect(self.actLoadImageClicked)
        self.ui.cbPosition.currentIndexChanged.connect(self.cbPositionIndexChanged)
        self.ui.pbShow.clicked.connect(self.pbShowClicked)
        self.ui.pbUpdate.clicked.connect(self.pbUpdateClicked)
        self.ui.pbCapture.clicked.connect(self.pbCaptureClicked)
        self.ui.pbOptimize.clicked.connect(self.pbOptimizeClicked)
        
        self.pbCaptureClicked()
        self.cbPositionIndexChanged()

        self.ui.sbPositionValue.setValue(self.camera.camera_shape[1 - self.ui.cbPosition.currentIndex()]//2)
        self.pbShowClicked()

    def showImageFileDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.HideNameFilterDetails
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Abrir imagem", "", "Imagem (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.webp *.pbm *.pgm *.ppm *.pxm *.pnm *.pfm *.sr *.ras *.tiff *.tif *.exr *.hdr *.pic);;Todos os arquivos (*)", options = options)        
        return filename

    def loadFigureGraphicsView(self, array, graphicsView):
        w = graphicsView.geometry().width()/matplotlib.rcParams["figure.dpi"]
        h = graphicsView.geometry().height()/matplotlib.rcParams["figure.dpi"]

        figure = Figure(figsize=(w, h))
        axes = figure.gca()
        axes.imshow(array, cmap='gray')

        canvas = FigureCanvas(figure)
        
        scene = QtWidgets.QGraphicsScene(graphicsView)
        scene.addWidget(canvas)
        graphicsView.setScene(scene)
    
    def showDialog(self, icon, title, message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(icon)
        msgBox.setText(message)
        msgBox.setWindowTitle(title)
        msgBox.exec()

    def cbPositionIndexChanged(self):
        self.ui.sbPositionValue.setMaximum(self.camera.camera_shape[1 - self.ui.cbPosition.currentIndex()])

    def actLoadImageClicked(self):
        filename = self.showImageFileDialog()
        if filename:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.image = image
            self.image_correction = image

            camera_image = self.camera.get_image()
            x_c, y_c = centroid(camera_image/255.)
            x_i, y_i = centroid(image/255.)

            scale = 1 if OVERWRITE else 2
            h, w = scale*np.array(image.shape)
            
            if np.any([h > SLM_SHAPE[0], w > SLM_SHAPE[1]]):
                self.showDialog(QtWidgets.QMessageBox.Warning, 'Aviso', 'Tamanho da imagem deve ser menor que a região do SLM (%d x %d)' % SLM_SHAPE)
                return
            
            y_max, x_max = SLM_SHAPE

            self.ui.sbX.setMaximum(x_max - w)
            self.ui.sbX.setValue(int((x_c*SLM_SHAPE[1]/self.camera.camera_shape[1] - scale*x_i)))
            self.ui.sbY.setMaximum(y_max - h)
            self.ui.sbY.setValue(int((y_c*SLM_SHAPE[0]/self.camera.camera_shape[0] - scale*y_i)))

            opt_args = (self.ui.sbX.value(), self.ui.sbY.value(), self.ui.dsbW.value(), self.ui.dsbA.value())
            phase = generate_pishift(image, opt_args = opt_args, overwrite = OVERWRITE, slm_shape = SLM_SHAPE, binary = BINARY)
            self.phase = phase
            self.camera.set_phase(phase)
            camera_image = self.camera.get_image()
    
            self.loadFigureGraphicsView(phase, self.ui.gvPhase)
            self.loadFigureGraphicsView(camera_image, self.ui.gvCamera)
            self.cbPositionIndexChanged()

            self.pbShowClicked()

            self.ui.gbPhase.setEnabled(True)
            self.ui.gbCameraSideView.setEnabled(True)
    
    def pbShowClicked(self):
        w = self.ui.gvCameraSideView.geometry().width()/matplotlib.rcParams["figure.dpi"]
        h = self.ui.gvCameraSideView.geometry().height()/matplotlib.rcParams["figure.dpi"]

        figure = Figure(figsize=(w, h))
        axes = figure.gca()
        camera_image = self.camera.get_image()
        x, y = None, None
        if self.ui.cbPosition.currentIndex() == 0:
            y = camera_image[:, self.ui.sbPositionValue.value() - 1]
        else:
            y = camera_image[self.ui.sbPositionValue.value() - 1, :]
        axes.set_xlim(1, len(y) + 1)
        x = np.arange(1, len(y) + 1)
        axes.set_ylim(0, 1.5*np.max(camera_image))
        axes.set_xlabel('Posicao')
        axes.set_ylabel('Intensidade')
        axes.plot(x, y)

        canvas = FigureCanvas(figure)
        
        scene = QtWidgets.QGraphicsScene(self.ui.gvCameraSideView)
        scene.addWidget(canvas)
        self.ui.gvCameraSideView.setScene(scene)

    def update(self, opt_args):
        self.phase = generate_pishift(self.image_correction, opt_args = opt_args, overwrite = OVERWRITE, slm_shape = SLM_SHAPE, binary = BINARY)
        self.camera.set_phase(self.phase)
        camera_image = self.camera.get_image()

        self.loadFigureGraphicsView(self.phase, self.ui.gvPhase)
        self.loadFigureGraphicsView(camera_image, self.ui.gvCamera)
        self.cbPositionIndexChanged()
        self.pbShowClicked()

    def pbUpdateClicked(self):
        opt_args = (self.ui.sbX.value(), self.ui.sbY.value(), None, None)
        self.update(opt_args)

    def pbCaptureClicked(self):
        self.loadFigureGraphicsView(self.camera.get_image(), self.ui.gvCamera)

    def pbOptimizeClicked(self):
        camera_image = self.camera.get_image()
        aligned_image = align_image(self.image, camera_image)
        self.image_correction = self.image - np.where(aligned_image > self.image, aligned_image - self.image, 0)
        
        opt_args = self.ui.sbX.value(), self.ui.sbY.value(), None, None
        self.update(opt_args)

app = QtWidgets.QApplication(sys.argv)

my_mainWindow = MainWindow()
my_mainWindow.show()

sys.exit(app.exec_())
