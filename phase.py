import numpy as np

def generate_pishift(image, line_thickness = 1):
    result = image.copy()
    for line in range(line_thickness):
        result[line::(2*line_thickness),:] = 128
    return result
