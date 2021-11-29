import numpy as np

def generate_pishift(image, line_thickness = 1, phase_max = 0.01):
    result = image.copy()
    result = (phase_max*128/np.pi)*result/np.max(result)
    for line in range(line_thickness):
        result[line::(2*line_thickness),:] = 128
    return result
