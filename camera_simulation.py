import cv2
import numpy as np

def generate_beam(shape, wl):
    height, width = shape
    I, J = np.arange(width), np.arange(height)
    I, J = np.meshgrid(I, J)
    i0, j0 = width/2., height/2.

    return np.exp(-((I-i0)**2.0 + (J-j0)**2.0)/(wl**2.0))

class Camera():
    def __init__(self, shape, monitor = 1):
        self.beam = generate_beam(shape, 350)
        self.field = self.beam
        self.camera_shape = shape

        self.monitor = monitor

    def set_phase(self, phase):
        field = self.beam*np.exp(1j*phase)
        field1 = np.fft.fft2(field)
        h, w = field1.shape
        x = np.arange(-w//2, w//2 + (w % 2))
        y = np.arange(-h//2, h//2 + (h % 2))
        x, y = np.meshgrid(x, y)
        size = 300
        field1 = (x**2 + y**2 <= size**2)*field1

        self.field = np.fft.fft2(field1)

    def get_image(self):
        image = np.abs(self.field)[::-1, ::-1]
        return (235*(image - np.min(image))/np.ptp(image) + 20).astype(np.uint8)
    
    def close(self):
        self.field = self.beam

