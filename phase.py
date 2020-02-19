import numpy as np

def generate_pishift(image, coords, shape, binary, grating_params = (1, 1)):
    """
    Calcula o padrao de fase usando a tecnica de contraste de fase generalizado de ordem zero
    
    Parameters
    ----------
    image: ndimage
        Imagem com o perfil de intensidade desejado
    coords: tuple (x0, y0)
        Lista de argumentos que visam a producao de potenciais mais lisos aplicando um filtro gaussiano inverso. Os parametros estao indicados na ordem 
            x0 - Posicao x da fase no canvas do SLM
            y0 - Posicao y da fase no canvas do SLM
        O padrao e None, em que o filtro nao e aplicado.
    shape: tuple
        Tamanho do padrao de fase
    binary: bool
        Indica se a imagem do potencial e binaria

    Returns
    -------
    ndarray
        Padrao de fase calculado
    """
    h_i, w_i = image.shape
    x0, y0 = coords

    g_l, g_a = grating_params

    imagen = (image - np.min(image))/np.ptp(image)
    phase = None

    if binary:
        phase = np.zeros_like(image)
        phase[np.nonzero(image)] = 128*imagen[np.nonzero(image)]
    else:
        phase = 128*np.sqrt(imagen)/np.pi
    
    h_p, w_p = phase.shape
    phase_slm = np.zeros(shape)
    phase_slm[y0:(y0 + h_p), x0:(x0 + w_p)] = phase

    for line in range(1, g_a + 1):
        phase_slm[line::(g_l + 1),:] = 128

    return phase_slm
