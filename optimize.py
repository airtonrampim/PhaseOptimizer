import cv2
import numpy as np

def flatness_cost(image):
    blur = cv2.GaussianBlur((255*image).astype(np.uint8),(5,5),0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    image_l = image[np.nonzero(mask)]
    M = np.sqrt(np.sum(image_l)/np.prod(image_l.shape))
    return np.sqrt(np.sum((image_l/M - 1)**2)/np.prod(image_l.shape))

def flatness_gradient(image, x0, y0, w, a):
    blur = cv2.GaussianBlur((255*image).astype(np.uint8),(5,5),0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    image_l = image[np.nonzero(mask)]

    M = np.sqrt(np.sum(image_l)/np.prod(image_l.shape))
    F = np.sqrt(np.sum((image_l/M - 1)**2)/np.prod(image_l.shape))

    height, width = image.shape

    x = np.arange(1, width + 1)
    y = np.arange(1, height + 1)
    X, Y = np.meshgrid(x, y)
    
    filter = 1 - a*np.exp(-((X - x0)**2 + (Y - y0)**2)/(2*w**2))
    filter_w = -((X - x0)**2 + (Y - y0)**2)/(w**3)*a*np.exp(-((X - x0)**2 + (Y - y0)**2)/(2*w**2))
    filter_a = -np.exp(-((X - x0)**2 + (Y - y0)**2)/(2*w**2))
    
    filter = filter[np.nonzero(mask)]
    filter_w = filter_w[np.nonzero(mask)]
    filter_a = filter_a[np.nonzero(mask)]

    grad_w = np.sum((image_l*filter/M - 1)*(image_l*filter_w - image_l*filter/(2*np.prod(image_l.shape)*M**2)*np.sum(image_l*filter_w)))/(np.prod(image_l.shape)*F*M)
    grad_a = np.sum((image_l*filter/M - 1)*(image_l*filter_a - image_l*filter/(2*np.prod(image_l.shape)*M**2)*np.sum(image_l*filter_a)))/(np.prod(image_l.shape)*F*M)

    return grad_w, grad_a

def centroid(image):
    h, w = image.shape
    X = np.arange(1, w + 1)
    Y = np.arange(1, h + 1)
    X, Y = np.meshgrid(X, Y)
    total = np.sum(image)
    x = np.sum(image*X)/total
    y = np.sum(image*Y)/total
    return x, y
