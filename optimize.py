import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


#https://stackoverflow.com/a/60064072/9257438
def get_corners(image, apply_threshold):    
    thresh_res = image
    if apply_threshold:
        gmm = GaussianMixture(n_components=2).fit(image.flatten().reshape(-1,1))

        mu1 = gmm.means_.flatten()[0]
        sigma1 = np.sqrt(gmm.covariances_.flatten()[0])
        mu2 = gmm.means_.flatten()[1]
        sigma2 = np.sqrt(gmm.covariances_.flatten()[1])    
        if mu1 > mu2:
            mu1 = gmm.means_.flatten()[1]
            sigma1 = np.sqrt(gmm.covariances_.flatten()[1])
            mu2 = gmm.means_.flatten()[0]
            sigma2 = np.sqrt(gmm.covariances_.flatten()[0])

        thresh_value = mu1
        if sigma1 == sigma2:
            if mu1 != mu2:
                thresh_value = (mu1+mu2)/2
        else:
            thresh_value = (sigma2**2*mu1-sigma1**2*mu2)/(sigma2**2-sigma1**2) + sigma1*sigma2/(sigma2**2-sigma1**2)*np.sqrt((mu2-mu1)**2+2*(sigma2**2-sigma1**2)*np.log(sigma2/sigma1))

        thresh_value, thresh = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        thresh_res = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(thresh_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    res_contour = contours[0]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if (area > area_thresh):
            area = area_thresh
            res_contour = c

    # get rotated rectangle from contour
    rot_rect = cv2.minAreaRect(res_contour)
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    return box

def get_warp(image, camera):
    image_box = get_corners(image, False)
    camera_box = get_corners(camera, True)
    warp, match_res = cv2.estimateAffine2D(camera_box, image_box)
    return warp, np.sum(match_res)/len(match_res)

def centroid(image):
    h, w = image.shape
    X = np.arange(1, w + 1)
    Y = np.arange(1, h + 1)
    X, Y = np.meshgrid(X, Y)
    total = np.sum(image)
    x = np.sum(image*X)/total
    y = np.sum(image*Y)/total
    return x, y
