import cv2
import numpy as np

def align_image(image, camera, affine_transform):
    """
    Ajusta a imagem da camera com base na imagem original da armadilha

    Parameters
    ----------
    image: ndarray
       Imagem original da armadilha
    camera: ndarray
       Imagem obtida pela camera

    Returns
    -------
    ndarray
       Imagem da camera reposicionada/redimensionada e rotacionada na posicao mais compativel com a imagem inicial
    """
    if affine_transform:
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

        warp, match_res = cv2.estimateAffinePartial2D(kl_image, kl_camera)

        #confidence = 1E2*np.nonzero(match_res)[0].size/match_res.size
        #print(r'Nivel de confidencia obtida para a transformada: %.2f%%' % confidence)
        
        return cv2.warpAffine(image, warp, camera.shape[::-1])
    else:
        camera_shape = np.array(camera.shape)
        camera_ratio = camera_shape//np.gcd(*camera.shape)
        image_shape = np.array(image.shape)
        image_ratio = image_shape//np.gcd(*image.shape)
        if np.all(camera_ratio == image_ratio):
            return cv2.resize(image, camera.shape[::-1])
        else:
            div_ratio = camera_shape // image_shape
            resized_image = image
            if np.all(div_ratio - 1): # Redimensionar a imagem
                m = np.min(div_ratio)
                resized_image = cv2.resize(image, (m*image_shape)[::-1])
            resized_image_shape = np.array(resized_image.shape)
            new_shape = np.max([camera_shape, resized_image_shape], axis=0)
            new_image = np.zeros(new_shape)
            
            putimg_i = np.abs(new_shape - resized_image_shape)//2 + (np.abs(new_shape - resized_image_shape)%2)
            putimg_f = new_shape - putimg_i
            
            new_image[putimg_i[0]:putimg_f[0], putimg_i[1]:putimg_f[1]] = resized_image
            
            cutimg_i = np.abs(new_shape - camera_shape)//2 + (np.abs(new_shape - camera_shape)%2)
            cutimg_f = new_shape - cutimg_i

            return new_image[cutimg_i[0]:cutimg_f[0], cutimg_i[1]:cutimg_f[1]]

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
