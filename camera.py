import numpy as np
import PyCapture2
import slmpy

class Camera():
    def __init__(self, shape, icon = None):
        bus = PyCapture2.BusManager()
        self.camera = PyCapture2.Camera()
        uid = bus.getCameraFromIndex(0)
        self.camera.connect(uid)
        self.camera.startCapture()
        self.slm = slmpy.SLMdisplay(shape, icon)
        image = self.camera.retrieveBuffer()
        self.camera_shape = (image.getRows(), image.getCols())

    def set_phase(self, phase):
        self.slm.updateArray((phase*120./255.).astype(np.uint8))

    def get_image(self):
        image = self.camera.retrieveBuffer()
        self.camera_shape = (image.getRows(), image.getCols())
        image_array = np.array(image.getData(), dtype='uint8').reshape(self.camera_shape)
        return image_array

    def close(self):
        self.slm.close()
