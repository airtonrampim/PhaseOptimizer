import numpy as np
from optimize import centroid

def generate_pishift(image, opt_args, slm_shape, overwrite, binary):
    """
    Calcula o padrao de fase usando a tecnica de contraste de fase generalizado de ordem zero
    
    Parameters
    ----------
    image: ndimage
        Imagem com o perfil de intensidade desejado
    opt_args: tuple (x0, y0, w, a), optional
        Lista de argumentos que visam a producao de potenciais mais lisos aplicando um filtro gaussiano inverso. Os parametros estao indicados na ordem 
            x0 - Posicao x da fase no canvas do SLM
            y0 - Posicao y da fase no canvas do SLM
            w  - Tamanho da cintura do filtro gaussiano inverso
            a  - Amplitude do filtro gaussiano inverso
        O padrao e None, em que o filtro nao e aplicado.
    slm_shape: tuple, optional
        Tamanho do SLM. O padrao e (1024, 1280)
    overwrite: bool, optional
        Indica se as linhas constantes $\pi$ serao sobrescritas na imagem, gerando um padrao de fase com o mesmo tamanho da imagem. O padrao e False
    
    Returns
    -------
    ndarray
        Padrao de fase calculado
    """
    inverse_filter = 1
    h_i, w_i = image.shape
    x0, y0, w, a = opt_args

    x = np.arange(1, w_i + 1)
    y = np.arange(1, h_i + 1)
    X, Y = np.meshgrid(x, y)
    xf, yf = centroid(image)
    inverse_filter = 1 - a*np.exp(-((X - xf)**2 + (Y - yf)**2)/(w**2))

    imagen = inverse_filter*image
    imagen = (imagen - np.min(imagen))/np.ptp(imagen)
    phase = None
    if overwrite:
        if binary:
            phase = np.zeros_like(image)
            phase[np.nonzero(image)] = 128*imagen[np.nonzero(image)]
        else:
            phase = 128*np.sqrt(imagen)/np.pi
        phase[1::2,:] = 128
    else:
        phase = 128*np.ones(2*np.array(imagen.shape))
        phase[0::2, 0::2] = 128*np.sqrt(imagen)/np.pi
        phase[0::2, 1::2] = phase[0::2, 0::2]
    
    h_p, w_p = phase.shape
    phase_slm = np.zeros(slm_shape)
    phase_slm[y0:(y0 + h_p), x0:(x0 + w_p)] = phase
    return phase_slm
