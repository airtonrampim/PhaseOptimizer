import cv2
import numpy as np

def generate_beam(shape, wl):
    height, width = shape
    I, J = np.arange(width), np.arange(height)
    I, J = np.meshgrid(I, J)
    i0, j0 = width/2., height/2.

    return np.exp(-((I-i0)**2.0 + (J-j0)**2.0)/(wl**2.0))

def generate_noise(coord, shape, wl):
    height, width = shape
    I, J = np.arange(width), np.arange(height)
    I, J = np.meshgrid(I, J)
    i0, j0 = coord
    return 1 - 0.5*np.exp(-((I-i0)**2.0 + (J-j0)**2.0)/(wl**2.0))

class Camera():
    def __init__(self, shape, monitor = 1):
        wl = 250
        self.beam = generate_noise((600, 550),shape,20)*generate_beam(shape, wl)
        
        self.field = self.beam.astype(np.complex128)
        self.camera_shape = shape

        self.monitor = monitor

    def set_phase(self, phase):
        field = self.beam*np.exp(1j*phase*np.pi/128)
        field_f = np.fft.fftshift(np.fft.fft2(field))
        h, w = field_f.shape
        x = np.arange(-w//2, w//2 + (w % 2))
        y = np.arange(-h//2, h//2 + (h % 2))
        x, y = np.meshgrid(x, y)
        x0, y0 = 0, 0
        size = 50
        P = ((x - x0)**2 + (y - y0)**2 <= size**2)
        field_f0 = P*field_f

        self.field = np.fft.ifft2(field_f0)

    def get_image(self):
        image = np.abs(self.field)
        return (255*(image - np.min(image))/np.ptp(image)).astype(np.uint8)
    
    def close(self):
        pass
        #self.field = self.beam.astype(np.complex128)
