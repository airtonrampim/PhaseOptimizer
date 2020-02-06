import cv2
import numpy as np

def get_corners(image):
    # blur image
    image_inv = 255 - image
    blur = cv2.GaussianBlur(image_inv, (3,3), 0)

    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 75, 2)

    # apply morphology
    kernel = np.ones((5,5), np.uint8)
    rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

    # thin
    kernel = np.ones((5,5), np.uint8)
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

    # get largest contour
    contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    big_contour = 0
    for c in contours:
        area_thresh = 0
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c

    # get rotated rectangle from contour
    rot_rect = cv2.minAreaRect(big_contour)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    return box

def align_image(image, camera, warp):
    mask = np.where(image > 0, 1, 0)
    return mask*cv2.warpAffine(camera, warp, image.shape[::-1])

def get_warp(image, camera):
    image_box = get_corners(image)
    camera_box = get_corners(camera)
    warp, match_res = cv2.estimateAffine2D(camera_box, image_box)
    return warp, np.sum(match_res)/len(match_res)

def get_warp2(image, camera):
    orb = cv2.ORB_create()
    kp_image, des_image = orb.detectAndCompute(image, None)
    kp_camera, des_camera = orb.detectAndCompute(camera, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    matches = sorted(bf.match(des_camera, des_image), key = lambda x: x.distance)
    
    kl_image, kl_camera = [], []
    
    for m in matches:
        kl_camera.append(kp_camera[m.queryIdx].pt)
        kl_image.append(kp_image[m.trainIdx].pt)
    
    kl_camera = np.float32(kl_camera)
    kl_image = np.float32(kl_image)
    
    warp, match_res = cv2.estimateAffine2D(kl_image, kl_camera)
    if warp is None:
        warp, match_res = cv2.estimateAffinePartial2D(kl_image, kl_camera)
    return warp, np.sum(match_res)/len(match_res)

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
