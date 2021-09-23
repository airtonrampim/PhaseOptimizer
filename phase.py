import numpy as np

def adjust_image(image, coords, shape):
    h_i, w_i = image.shape
    x0, y0 = coords

    result = np.zeros(shape)
    result[y0:(y0 + h_i), x0:(x0 + w_i)] = image

    return result

def generate_pishift(image, line_thickness = 1):
    result = image.copy()
    result = 0.41*result/np.max(result)
    for line in range(line_thickness):
        result[line::(2*line_thickness),:] = 128
    return result
