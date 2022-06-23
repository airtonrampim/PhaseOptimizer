import cv2
import time
import numpy as np

def generate_beam(shape, wl):
    height, width = shape
    I, J = np.arange(width), np.arange(height)
    I, J = np.meshgrid(I, J)
    i0, j0 = width/2., height/2.

    return 150*np.exp(-((I-i0)**2.0 + (J-j0)**2.0)/(wl**2.0))

class Camera():
    def __init__(self, shape, monitor = 1):
        self.wl = 150
        self.shape = shape
        self.camera_shape = (480, 640)
        self.phase = np.zeros(self.shape)
        self.beam = generate_beam(self.shape, self.wl)
        self.pert = 1 - 0.15*np.random.random(size=self.camera_shape)

        h, w = self.camera_shape
        x = np.arange(-w//2, w//2 + (w % 2))
        y = np.arange(-h//2, h//2 + (h % 2))
        x, y = np.meshgrid(x, y)

        self.pert_static = 1 - 0.3*np.abs(np.sin(y*np.pi/65))**1.5
        self.time = time.time()

        self.field = self.beam.astype(np.complex128)

        self.monitor = monitor

    def set_phase(self, phase):
        self.phase = phase
        field = self.beam*np.exp(1j*self.phase*np.pi/128)
        field_f = np.fft.fftshift(np.fft.fft2(field))
        h, w = field_f.shape
        x = np.arange(-w//2, w//2 + (w % 2))
        y = np.arange(-h//2, h//2 + (h % 2))
        x, y = np.meshgrid(x, y)
        x0, y0 = 0, 0
        size = 150
        P = ((x - x0)**2 + (y - y0)**2 <= size**2)
        field_f0 = P*field_f

        self.field = np.fft.ifft2(field_f0)

    def get_image(self):
        now = time.time()
        if now - self.time > 0.1:
            self.pert = 1 - 0.15*np.random.random(size=self.camera_shape)
        image = np.abs(self.field)
        image_l = cv2.warpAffine(image, np.array([[1.2*np.cos(0.05), -1.2*np.sin(0.05), -350], [1.2*np.sin(0.05), 1.2*np.cos(0.05), -400]], dtype=np.float64), self.camera_shape[::-1])*self.pert_static*self.pert
        return np.floor(image_l).astype(np.uint8) + 40

    def close(self):
        pass
        #self.field = self.beam.astype(np.complex128)
