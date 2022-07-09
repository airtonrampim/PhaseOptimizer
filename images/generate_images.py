import os
import cv2
import numpy as np

# -71, 36
X0, Y0 = 565, 548
PI_VALUE = 94
PATH = 'generated_images'

if not os.path.exists(PATH):
    os.mkdir(PATH)
zeros = np.zeros((1024, 1272), dtype = np.uint8)

# Background
cv2.imwrite(os.path.join(PATH, 'background.png'), zeros)

# Rectangle
rect = zeros.copy()
rect[(Y0 - 75):(Y0 + 75), (X0 - 110):(X0 + 110)] = PI_VALUE
cv2.imwrite(os.path.join(PATH, 'rectangle.png'), rect)

# Ramp
ramp_1px = zeros.copy()
ramp_1px[(Y0 - 64):(Y0 + 64), (X0 - 128):(X0 + 128)] = np.arange(0, 256, 1).astype(np.uint8)
cv2.imwrite(os.path.join(PATH, 'rampa_1px.png'), ramp_1px)

ramp_2px = zeros.copy()
ramp_2px[(Y0 - 128):(Y0 + 128), (X0 - 256):(X0 + 256)] = np.floor(np.arange(0, 256, 0.5)).astype(np.uint8)
cv2.imwrite(os.path.join(PATH, 'rampa_2px.png'), ramp_2px)

ramp_1px_ad = zeros.copy()
ramp_1px_ad[(Y0 - 128):(Y0 + 128), (X0 - 256):(X0 + 256)] = np.hstack((np.arange(0, 256, 1), np.arange(255, -1, -1))).astype(np.uint8)
cv2.imwrite(os.path.join(PATH, 'rampa_1px_ad.png'), ramp_1px_ad)

# Pixels
if not os.path.exists(os.path.join(PATH, 'pixels')):
    os.mkdir(os.path.join(PATH, 'pixels'))
for px in np.arange(1, 11, 1):
    pixel_image = zeros.copy()
    pixel_image[(Y0 - px):(Y0 + px), (X0 - (px // 2)):(X0 + (px // 2) + (px % 2))] = PI_VALUE
    cv2.imwrite(os.path.join(PATH, 'pixels', '%dpx.png' % px), pixel_image)
