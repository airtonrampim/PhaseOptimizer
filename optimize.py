import cv2
import numpy as np
#from sklearn.mixture import GaussianMixture

def avgpool(array, factor):
    array_avgpool = np.zeros((array.shape[0] // (2*factor), array.shape[1] // factor), dtype=np.float64)
    np.add.at(array_avgpool, (np.arange(array.shape[0])[:, np.newaxis] // (2*factor), np.arange(array.shape[1]) // factor), array)
    return np.round(array_avgpool/(2*factor**2)).astype(array.dtype)

def maxpool(array, factor):
    array_maxpool = np.full((array.shape[0] // (2*factor), array.shape[1] // factor), 0, dtype=array.dtype)
    np.maximum.at(array_maxpool, (np.arange(array.shape[0])[:, np.newaxis] // (2*factor), np.arange(array.shape[1]) // factor), array)
    return array_maxpool

def collapse_array(array, factor):
    return avgpool(array, factor)

def expand_array(array, factor):
    return np.repeat(np.repeat(array, 2*factor, axis = 0), factor, axis = 1)

def find_phase_subregion(phase, factor):
    i, j = np.where(phase > 0)
    imin, imax = np.min(i), np.max(i) + 1
    jmin, jmax = np.min(j), np.max(j) + 1

    # Resize the region to a multiple of factor for each axis
    iresize = (imax - imin) % (2*factor)
    if iresize != 0:
        iresize = 2*factor - iresize
    jresize = (jmax - jmin) % factor
    if jresize != 0:
        jresize = factor - jresize
    imin_l = imin - (iresize // 2)
    imax_l = imax + (iresize // 2) + (iresize % 2) - 1
    jmin_l = jmin - (jresize // 2)
    jmax_l = jmax + (jresize // 2) + (jresize % 2) - 1

    return imin_l, imax_l, jmin_l, jmax_l

#https://stackoverflow.com/a/60064072/9257438
def get_corners(image, apply_threshold):    
    thresh_res = image
    if apply_threshold:
        #gmm = GaussianMixture(n_components=2).fit(image.flatten().reshape(-1,1))

        #mu1 = gmm.means_.flatten()[0]
        #sigma1 = np.sqrt(gmm.covariances_.flatten()[0])
        #mu2 = gmm.means_.flatten()[1]
        #sigma2 = np.sqrt(gmm.covariances_.flatten()[1])    
        #if mu1 > mu2:
            #mu1 = gmm.means_.flatten()[1]
            #sigma1 = np.sqrt(gmm.covariances_.flatten()[1])
            #mu2 = gmm.means_.flatten()[0]
            #sigma2 = np.sqrt(gmm.covariances_.flatten()[0])

        #thresh_value = mu1
        #if sigma1 == sigma2:
            #if mu1 != mu2:
                #thresh_value = (mu1+mu2)/2
        #else:
            #thresh_value = (sigma2**2*mu1-sigma1**2*mu2)/(sigma2**2-sigma1**2) + sigma1*sigma2/(sigma2**2-sigma1**2)*np.sqrt((mu2-mu1)**2+2*(sigma2**2-sigma1**2)*np.log(sigma2/sigma1))

        #thresh_value, thresh = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
        thresh_value, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    image_box = image_box[np.argsort(np.sum(image_box**2, axis = 1))]
    camera_box = camera_box[np.argsort(np.sum(camera_box**2, axis = 1))]

    warp, match_res = cv2.estimateAffine2D(camera_box, image_box)
    return warp, np.sum(match_res)/len(match_res)

def get_warp_inverse(warp): 
    return np.array([[warp[1,1], -warp[0,1], warp[0,1]*warp[1,2] - warp[1,1]*warp[0,2]], [-warp[1,0], warp[0,0], warp[1,0]*warp[0,2] - warp[0,0]*warp[1,2]]])/(warp[0,0]*warp[1,1] - warp[0,1]*warp[1,0])

def centroid(image):
    h, w = image.shape
    X = np.arange(1, w + 1)
    Y = np.arange(1, h + 1)
    X, Y = np.meshgrid(X, Y)
    total = np.sum(image)
    x = np.sum(image*X)/total
    y = np.sum(image*Y)/total
    return x, y
