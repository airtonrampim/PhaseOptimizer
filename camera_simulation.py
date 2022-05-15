import cv2
import time
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
        self.wl = 250
        #self.beam = generate_noise((600, 550),shape,20)*generate_beam(shape, wl)
        self.phase = np.zeros(shape)
        self.beam = generate_beam(shape, self.wl)
        self.time = time.time()
        
        self.field = self.beam.astype(np.complex128)
        self.camera_shape = (480, 640)

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
        size = 50
        P = ((x - x0)**2 + (y - y0)**2 <= size**2)
        field_f0 = P*field_f

        self.field = np.fft.ifft2(field_f0)

    def get_image(self):
        #now = time.time()
        #if now - self.time > 0.1:
            #self.beam = generate_beam(self.camera_shape, self.wl)*(1 - np.random.normal(size=self.camera_shape))
            #self.set_phase(self.phase)
            #self.time = now
        image = np.abs(self.field)
        beam_amp = np.max(self.beam)
        image_l = cv2.warpAffine(240*image**2/beam_amp**2, np.array([[0.7*np.cos(0.05), 0.7*np.sin(0.05), -100], [-0.7*np.sin(0.05), 0.7*np.cos(0.05), -50]], dtype=np.float64), self.camera_shape[::-1])
        return np.floor(image_l).astype(np.uint8)
    
    def close(self):
        pass
        #self.field = self.beam.astype(np.complex128)
