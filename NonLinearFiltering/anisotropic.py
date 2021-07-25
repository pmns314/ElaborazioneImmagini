from scipy import stats
from skimage.restoration import estimate_sigma
from NonLinearFiltering.Morphological import morphological_open, morphological_close
from Utils import *


def anisodiff(img, kappa=0.2, option=1, neightborhood='minimal'):
    """
    Realizza il filtraggio anisotropico di un immagine effettuando una singola iterazione
    :param img : immagine in scala di grigio di cui si vuole eseguire il filtraggio anisotropico
    :param kappa: soglia del gradiente
    :param option: scelta della funzione di conducibilità
    * Se option = 1 viene scelta come funzione di couducibilità quella esponenziale
    * Se option = 2 viene scelta come funzione di couducibilità quella quadratica
    :param neightborhood:
     * Se neightborhood = 'minimal' l'algoritmo considera solo delle variazioni dell'immagine lungo le direzioni {N,S,E,W}
     * Se neightborhood = 'maximal' l'algoritmo considera delle variazioni dell'immagine lungo le direzioni {N,S,E,W,NE,NW,SE,SW}
    :return: immagine filtrata
    """
    imgout = im2double(img.copy())

    if neightborhood == 'minimal':

        gamma = 1 / 4  # rate di diffusione, questo parametro dipende dalle direzioni considerate

        # matrici che che conterranno i valori restituiti dalla funzione di conducibilità scelta
        gS = np.zeros_like(imgout)
        gN = gS.copy()
        gE = gS.copy()
        gW = gS.copy()

        # matrici che che conterranno le variazioni nell 4 direzioni {N,S,E,W}
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        deltaN = deltaS.copy()
        deltaW = deltaS.copy()

        # calcolo le differenze lungo le 4 direzione
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
        deltaN[1:, :] = -deltaS[:-1, :]
        deltaW[:, 1:] = -deltaE[:, :-1]

        # scelta della funzione di conducibilità
        if option == 1:
            # equazione esponenziale
            gS = np.exp(-(np.abs(deltaS) / np.abs(kappa)) ** 2)
            gE = np.exp(-(np.abs(deltaE) / np.abs(kappa)) ** 2)
            gN = np.exp(-(np.abs(deltaN) / np.abs(kappa)) ** 2)
            gW = np.exp(-(np.abs(deltaW) / np.abs(kappa)) ** 2)



        elif option == 2:
            # equazione quadratica
            gS = 1 / (1 + (np.abs(deltaS) / kappa) ** 2)
            gE = 1 / (1 + (np.abs(deltaE) / kappa) ** 2)
            gN = 1 / (1 + (np.abs(deltaS) / kappa) ** 2)
            gW = 1 / (1 + (np.abs(deltaW) / kappa) ** 2)

        # calcolo dei pesi
        N = gN * deltaN
        S = gS * deltaS
        W = gW * deltaW
        E = gE * deltaE

        # aggiornamento dell'immagine
        imgout += gamma * (N + S + E + W)

        return imgout

    elif neightborhood == 'maximal':

        gamma = 1 / 8

        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        deltaN = deltaS.copy()
        deltaW = deltaS.copy()

        gS = np.zeros_like(imgout)
        gN = gS.copy()
        gE = gS.copy()
        gW = gS.copy()
        gNE = gS.copy()
        gNW = gS.copy()
        gSE = gS.copy()
        gSW = gS.copy()

        # kernel diagonali
        win_NE = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]])
        win_NW = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        win_SE = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
        win_SW = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])

        # calcolo delle differenze lungo le 8 direzioni {N,S,E,W,NE,NW,SE,SW}
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
        deltaN[1:, :] = -deltaS[:-1, :]
        deltaW[:, 1:] = -deltaE[:, :-1]
        deltaNE = cv2.filter2D(imgout, -1, win_NE)
        deltaNW = cv2.filter2D(imgout, -1, win_NW)
        deltaSE = cv2.filter2D(imgout, -1, win_SE)
        deltaSW = cv2.filter2D(imgout, -1, win_SW)

        if option == 1:

            gS = np.exp(-(np.abs(deltaS) / np.abs(kappa)) ** 2)
            gE = np.exp(-(np.abs(deltaE) / np.abs(kappa)) ** 2)
            gN = np.exp(-(np.abs(deltaN) / np.abs(kappa)) ** 2)
            gW = np.exp(-(np.abs(deltaW) / np.abs(kappa)) ** 2)
            gNE = np.exp(-(np.abs(deltaNE) / np.abs(kappa)) ** 2)
            gNW = np.exp(-(np.abs(deltaNW) / np.abs(kappa)) ** 2)
            gSE = np.exp(-(np.abs(deltaSE) / np.abs(kappa)) ** 2)
            gSW = np.exp(-(np.abs(deltaSW) / np.abs(kappa)) ** 2)


        elif option == 2:

            gS = 1 / (1 + (np.abs(deltaS) / kappa) ** 2)
            gE = 1 / (1 + (np.abs(deltaE) / kappa) ** 2)
            gN = 1 / (1 + (np.abs(deltaS) / kappa) ** 2)
            gW = 1 / (1 + (np.abs(deltaW) / kappa) ** 2)
            gNE = 1 / (1 + (np.abs(deltaNE) / kappa) ** 2)
            gNW = 1 / (1 + (np.abs(deltaNW) / kappa) ** 2)
            gSE = 1 / (1 + (np.abs(deltaSE) / kappa) ** 2)
            gSW = 1 / (1 + (np.abs(deltaSW) / kappa) ** 2)

        N = gN * deltaN
        S = gS * deltaS
        W = gW * deltaW
        E = gE * deltaE
        NE = gNE * deltaNE
        NW = gNW * deltaNW
        SE = gSE * deltaSE
        SW = gSW * deltaSW

        imgout += gamma * (N + S + E + W + NE + NW + SE + SW)
    return imgout


def aniso(image, number_iteration, kappas, option, neightborhood, single_threshold):
    """
        Realizza il filtraggio anisotropico di un immagine considerando un numero specifico di iterazioni
        :param image: immagine in scala di grigio di cui si vuole eseguire il filtraggio anisotropico
        :param kappas: soglia del gradiente
        :param option: scelta della funzione di conducibilità
        * Se option = 1 viene scelta come funzione di couducibilità quella esponenziale
        * Se option = 2 viene scelta come funzione di couducibilità quella quadratica
        :param neightborhood:
         * Se neightborhood = 'minimal' l'algoritmo considera solo delle variazioni dell'immagine lungo le direzioni {N,S,E,W}
         * Se neightborhood = 'maximal' l'algoritmo considera delle variazioni dell'immagine lungo le direzioni {N,S,E,W,NE,NW,SE,SW}
         :param single_threshold:
         * Se single_threshold = True: utilizza un'unica soglia
         * se single_threshold = False: utilizza più soglie diverse ad ogni iterazione
        :return: immagine filtrata
        """
    I = image.copy()
    # più è alto il numero di iterazioni più l'immagine risulta sfocata
    if not single_threshold:
        for i in range(number_iteration):
            I = anisodiff(I, kappas[i], option, neightborhood)
    else:
        for i in range(number_iteration):
            I = anisodiff(I, kappas, option, neightborhood)
    return I


def anisoRGB(image, it_R, it_G, it_B, k_R, k_G, k_B, option, neightborhood, single_threshold):
    """
    Filtraggio anisotropico su un immagine RGB
    :param image: immagine RGB
    :param it_R: numero di iterazioni sul canale rosso
    :param it_G: numero di iterazioni sul canale verde
    :param it_B: numero di iterazioni sul canale blu
    :param k_R: soglia del gradiente sul canale rosso
    :param k_G: soglia del gradiente sul canale verde
    :param k_B: soglia del gradiente sul canale blu
    :param option: scelta della funzione di conducibilità
    * Se option = 1 viene scelta come funzione di couducibilità quella esponenziale
    * Se option = 2 viene scelta come funzione di couducibilità quella quadratica
    :param neightborhood:
     * Se neightborhood = 'minimal' l'algoritmo considera solo delle variazioni dell'immagine lungo le direzioni {N,S,E,W}
     * Se neightborhood = 'maximal' l'algoritmo considera delle variazioni dell'immagine lungo le direzioni {N,S,E,W,NE,NW,SE,SW}
    :param single_threshold:
    * Se single_threshold = True: utilizza un'unica soglia per canale
    * se single_threshold = False: utilizza più soglie per ogni canale
    :return immagine filtrata:
    """
    R = aniso(image[:, :, 0], it_R, k_R, option, neightborhood, single_threshold)
    G = aniso(image[:, :, 1], it_G, k_G, option, neightborhood, single_threshold)
    B = aniso(image[:, :, 2], it_B, k_B, option, neightborhood, single_threshold)
    return np.dstack((R, G, B))


def anisotropic_denoising(image, iterations=None, kappas=None, option=1, neighbourhood='minimal',
                          single_threshold=True):
    """
    Filtraggio anisotropico su un'immagine
    :param image:  immagine da filtrare
    :param iterations: numero di iterazioni. Può essere un intero o una lista di 3 elementi, uno per ogni canale
                       dell'immagine a colori.Opzionale: se non specificato, verrà calcolato in automatico
    :param kappas: soglia del gradiente sul canale. Può essere un intero o una lista di 3 elementi, uno per ogni canale
                   dell'immagine a colori.Opzionale, se non specificato verrà calcolato in automatico
    :param option: scelta della funzione di conducibilità
    * Se option = 1 viene scelta come funzione di couducibilità quella esponenziale
    * Se option = 2 viene scelta come funzione di couducibilità quella quadratica
    :param neighbourhood:
     * Se neighbourhood = 'minimal' l'algoritmo considera solo delle variazioni dell'immagine lungo le direzioni {N,S,E,W}
     * Se neighbourhood = 'maximal' l'algoritmo considera delle variazioni dell'immagine lungo le direzioni {N,S,E,W,NE,NW,SE,SW}
    :param single_threshold:
    * Se single_threshold = True: utilizza un'unica soglia per canale
    * se single_threshold = False: utilizza più soglie per ogni canale
    :return immagine filtrata:
    """

    if iterations is None and kappas is not None or iterations is not None and kappas is None:
        raise Exception(' Error: iterations or kappas not found')
    automatic = True if iterations is None and kappas is None else False
    if get_channels_number(image) == 1:
        if automatic:
            return aniso(image, *automatic_parameters(image, option, neighbourhood), option, neighbourhood,
                         False)
        else:
            return aniso(image, iterations, kappas, option, neighbourhood, single_threshold)
    else:
        if automatic:
            it_list, kappa_list = automatic_parameters_RGB(image, option, neighbourhood)
            return anisoRGB(image, *it_list, *kappa_list, option, neighbourhood, False)
        else:
            if type(iterations) is int and type(kappas) is float:
                iterations = [iterations for _ in range(3)]
                kappas = [kappas for _ in range(3)]
                return anisoRGB(image, *iterations, *kappas, option, neighbourhood, single_threshold)
            elif len(iterations) == 3 and len(kappas) == 3:
                return anisoRGB(image, *iterations, *kappas, option, neighbourhood, single_threshold)
            else:
                raise Exception("Wrong parameters!")


def automatic_parameters_RGB(image, option, neigthborhood):
    """
    Calcola automaticamente i parametri utilizzati dal filtro anisotropico per un immagine RGB
    :param image: immagine RGB
    :param option: opzione del filtro anisotropico
    :param neigthborhood: direzioni considerate dal filtro anisotropico
    :return: soglia del gradiente e numero di iterazioni per ogni rispettivo canale RGB
    """
    R_param = automatic_parameters(image[:, :, 0], option, neigthborhood)
    G_param = automatic_parameters(image[:, :, 1], option, neigthborhood)
    B_param = automatic_parameters(image[:, :, 2], option, neigthborhood)
    return [R_param[0], G_param[0], B_param[0]], [R_param[1], G_param[1], B_param[1]]


def get_gradient_thresh_MAD_RGB(image):
    """
    Stima la soglia del gradiente per ogni canale RGB dell'immagine usando la deviazione assoluta mediana (MAD)
    :param image: immagine di cui si vuole calcolare le soglie
    :return: le soglie per ogni canale
       """
    thresh_R = get_gradient_thresh_MAD(image[:, :, 0])
    thresh_G = get_gradient_thresh_MAD(image[:, :, 1])
    thresh_B = get_gradient_thresh_MAD(image[:, :, 2])
    return thresh_R, thresh_G, thresh_B


def get_gradient_thresh_morpho_RGB(image):
    """
    Stima la soglia del gradiente per ogni canale RGB dell'immagine usando un approccio morfologico
    :param image: immagine di cui si vuole calcolare le soglie
    :return: la soglie per ogni canale
       """
    thresh_R = get_gradient_thresh_morpho(image[:, :, 0])
    thresh_G = get_gradient_thresh_morpho(image[:, :, 1])
    thresh_B = get_gradient_thresh_morpho(image[:, :, 2])
    return thresh_R, thresh_G, thresh_B


def get_gradient_threshold(image):
    """
    Stima la soglia del gradiente sfruttando l'algoritmo proposto da Perona-Malik
    :param image: immagine di cui si vuole calcolare la soglia del gradiente
    :return: la soglia del gradiente
    """
    # gx = cv2.Sobel(image, -1, 1, 0)
    # gx = np.abs(gx)
    # gx = gx / np.max(gx)  # scaling
    # gx = sk.img_as_ubyte(gx)
    # hist = cv2.calcHist([gx], [0], None, [256], (0, 256))
    grad = gradient_magnitude(image)
    grad = grad / np.max(grad)
    grad = sk.img_as_ubyte(grad)
    hist = cv2.calcHist([grad], [0], None, [256], (0, 256))
    prob = hist / (image.shape[0] * image.shape[1])
    cdf = np.cumsum(prob)
    temp = np.where(
        cdf > 0.9)  # le variazioni che vanno oltre il 90% della distribuzione del gradiente devono essere preservati => pixel valutati come edge
    grad_tresh = temp[0][0] / (len(hist) - 1)  # scaling
    return grad_tresh


def get_gradient_thresh_morpho(image, kernel_size=5):
    """
    Stima la soglia del gradiente  usando un approccio morfologico
    :param image: immagine di cui si vuole calcolare la soglia del gradiente
    :param kernel_size: specifica la dimensione del kernel dell'elemento strutturante
    :return: la soglia del gradiente
    """
    kernel = np.ones([kernel_size, kernel_size])
    op = morphological_open(image, kernel)
    cl = morphological_close(image, kernel)
    sum_op = np.sum(op) / (image.shape[0] * image.shape[1])
    sum_cl = np.sum(cl) / (image.shape[0] * image.shape[1])
    grad_tresh = np.abs(sum_op - sum_cl)
    return grad_tresh


def get_gradient_thresh_MAD(image):
    """
    Stima la soglia del gradiente usando la deviazione assoluta mediana (MAD)
    :param image: immagine di cui si vuole calcolare la soglia del gradiente
    :return: la soglia del gradiente
    """
    grad = gradient_magnitude(image)
    MAD = 1.4826 * stats.median_abs_deviation(grad, axis=None)
    return MAD


def estimate_noise(image, multichannel=False, average_sigma=False):
    return estimate_sigma(image, multichannel, average_sigma)


def gradient_magnitude(image):
    """
    Calcola il gradiente di un immagine
    :param image: immagine di cui si vuole calcolare il gradiente:
    :return: gradiente dell'immagine
    """
    I = im2double(image.copy())
    gx, gy = gradient_xy(I)
    return np.sqrt(gx ** 2 + gy ** 2)


def gradient_xy(image):
    """
    Calcola le componenti gx e gy del gradiente
    :param image: immagine di cui si vuole calcolare gx e gy
    :return: gx e gy
    """
    I = im2double(image.copy())
    gx = cv2.Sobel(I, -1, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(I, -1, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return gx, gy


def find_edges_coordinates(image, num_edges, thresh):
    """
    Individua le coordinate di un numero di edge, caratterizzati dal più grande magnitude, che si trovano ad una distanza superiore ad una certa soglia
    :param image: immagine di cui si vuole calcolare gli edge
    :param num_edges: numero di edge da calcolare
    :param thresh: definisce una distanza minima tra gli edge
    :return: coordinate degli edge
    """
    edges_coordinates = np.zeros((num_edges, 2), dtype=np.uint8)  # lista delle coordinate
    grad_mag = gradient_magnitude(image)
    grad_m = np.zeros_like(grad_mag)
    row, col = grad_mag.shape

    # non considero i bordi del gradiente in modo da non tener conto degli edge sul bordo
    grad_m[4:-4, 4:-4] = grad_mag[4:-4, 4:-4]

    # riordinamento in ordine discendente i valori del gradiente
    sort_edge = np.reshape(grad_m, (1, row * col))[0]
    sort_edge[::-1].sort()

    # individuo le coordinate del valore più alto nel gradiente
    temp = grad_mag == sort_edge[0]
    edge_x1, edge_y1 = np.argwhere(temp)[0]
    edges_coordinates[:, 0] = edge_x1
    edges_coordinates[:, 1] = edge_y1

    i = 1  # contatore del numero di elementi nel gradiente
    j = 1  # contatore del numero di edge point

    # confronto delle distanze tra ogni coppia di edge
    while j <= num_edges - 1 and i <= row * col - 1:
        temp = grad_mag == sort_edge[i]
        edge_x1, edge_y1 = np.argwhere(temp)[0]
        edge = np.zeros((1, 2))  # edge da confrontare
        edge[0][0] = edge_x1
        edge[0][1] = edge_y1
        rep = np.tile(edge, (num_edges, 1))
        distance = np.sqrt(np.sum((rep - edges_coordinates) ** 2))  # distanza euclidea

        if distance > thresh:
            # aggiornamento delle coordinate
            edges_coordinates[j, :] = edge
            j += 1
        i += 1
    return edges_coordinates


def find_interpixel_location(image, edge_coordinates, sigma):
    """
    Individua le due regioni di interpixel per ogni edge
    :param image: immagine
    :param edge_coordinates: coordinate degli edge
    :param sigma: deviazione standard dell'immagine
    :return: il parametro alfa utilizzato per valutare la qualità degli edge e le due regioni per ogni edge point
    """
    num_inter_pixels = 12
    gx, gy = gradient_xy(image)
    phase = np.arctan2(gx, gy)
    num_edges = len(edge_coordinates)
    collection_regions = {}
    plot_edge_region = {}
    u0 = 0
    for i in range(num_edges):
        edge_x, edge_y = edge_coordinates[i]
        p = phase[edge_x][edge_y]
        inter_pixel_x = np.zeros((1, num_inter_pixels))
        inter_pixel_y = np.zeros((1, num_inter_pixels))

        # individuo le locazioni dei 12 interpixel
        z = 0
        x = np.arange(-1, 2)
        y = np.arange(-2, 3)
        for n in x:
            for m in y:
                if not m == 0:
                    inter_pixel_x[0][z] = edge_x + m * np.cos(p) - n * np.sin(p)
                    inter_pixel_y[0][z] = edge_y - m * np.sin(p) - n * np.cos(p)
                    z += 1

        plot_edge_region[i] = (inter_pixel_x[0], inter_pixel_y[0], [edge_x, edge_y])

        # uso l'interpolazione bilineare in quanto non conosciamo l'intensità degli  edge individuati nelle 2 regioni(le coordinate possono essere float)
        intensity_inter_pixel = np.zeros([1, num_inter_pixels])
        for k in range(num_inter_pixels):
            """
            interpolazione bilineare non lineare: 
            f(x,y) = a11 + a21*x +a12*y + a22*x*y
            a11 = f(1,1) 
            a21 = f(2,1) - f(1,1)
            a12 = f(1,2) - f(1,1)
            a22 = f(2,2) + f(1,1) - f(2,1) - f(1,2)
            """

            temp_i = int(inter_pixel_x[0][k])  # coordinata x del pixel più vicino all'edge
            temp_j = int(inter_pixel_y[0][k])  # coordinata y del pixel più vicino all'edge
            x = inter_pixel_x[0][k] - temp_i  # distanza rispetto alle x con il pixel più vicino
            y = inter_pixel_y[0][k] - temp_j  # distanza rispetto alle y con il pixel più vicino
            a11 = image[temp_i][temp_j]
            a21 = image[temp_i + 1][temp_j]
            a12 = image[temp_i][temp_j + 1] - image[temp_i][temp_j]
            a22 = image[temp_i + 1][temp_j + 1] + image[temp_i][temp_j] - (
                    image[temp_i + 1][temp_j] + image[temp_i][temp_j + 1])
            intensity_inter_pixel[0][
                k] = a11 + a21 * x + a12 * y + a22 * x * y  # lista delle intensità dei pixel individuati

        """
         regione1 => m < 0
         regione2 => m > 0
         i pixel che appartengono alla regione1(S) e nella regione2(N) si trovano nelle posizioni della lista intensity_inter_pixel: S S N N S S N N S S N N  
         quindi per facilitarmi il compito di individuare le intensità che appartengono ad una regione modello la lista in una matrice
          
          S S N N 
          S S N N
          S S N N
         in questo modo le prime due colonne rappresentano le intensità della prima regione e le ultime due colonne le intensità della seconda regione
          
        """
        # costruisco la matrice
        temp = np.zeros([3, 4])
        temp[0, :] = intensity_inter_pixel[0][0:4]
        temp[1, :] = intensity_inter_pixel[0][4:8]
        temp[2, :] = intensity_inter_pixel[0][8:12]

        # definisco le due regioni
        region_1 = temp[:, 0:2]
        region_2 = temp[:, 2:4]
        collection_regions[i] = (region_1, region_2)
        u0 += np.abs(np.mean(region_1) - np.mean(region_2))  # stima dell'intensità dell'edge

    u0 = u0 / num_edges
    alfa = sigma * 10 / u0  # stima dell'effetto del rumore sugli edge dell'immagine
    return alfa, collection_regions


def automatic_parameters(image, option, neightborhood):
    """
    Calcola automaticamente i parametri utilizzati dal filtro anisotropico
    :param image: immagine
    :param option: opzioni del filtro anisotropico
    :param neightborhood: direzioni considerate dal filtro anisotropico
    :return: soglia del gradiente e numero di iterazioni
    """
    max_iteration = 100
    num_edge = 200
    thresh = 1 / 35 * min(image.shape[0], image.shape[1])  # parametro  letteratura
    I = image.copy()
    Q = np.zeros([max_iteration, num_edge])
    current_iteration = 1
    kappas = []
    while True:

        kappa = get_gradient_threshold(I)  # calcolo ad ogni iterazione la soglia del gradiente
        kappas.append(kappa)
        I = anisodiff(I, kappa=kappa, option=option, neightborhood=neightborhood)
        sigma = estimate_noise(I)
        gaussian = cv2.getGaussianKernel(5, sigma)
        filt_gauss = cv2.filter2D(I, -1, gaussian)
        # applico un filtro gaussiano in quando il gradiente è contaminato dal rumore
        # in questo modo posso individuare meglio gli edge all'interno dell'immagine
        edge_coordinates = find_edges_coordinates(filt_gauss, num_edge, thresh)
        alfa, collection_regions = find_interpixel_location(I, edge_coordinates, sigma)
        """
        
        T = argmax(t) 1/N *sommatoria(Qi(t))  i=1,2...N
        Il parametro Q rappresenta la qualità dell'edge 
        
        Per implementare questa equazione costruisco una matrice che ha come coordinate l'iterazione corrente (riga) e il numero di edge considerato (colonna) 
        N = num_edge 
        M = max_iteration 
        0 Q1(0) Q2(0) Q3(0)...QN(0)
        1 Q1(1) Q2(1) Q3(1)...QN(1)
        .
        .
        .
        M
        
        """

        for i in range(num_edge):
            region1, region2 = collection_regions[i]
            Q[current_iteration - 1][i] = np.abs(np.mean(region1) - np.mean(region2)) - alfa * (
                    np.std(region1) + np.std(region2))

        if current_iteration > 1:
            average = Q.sum(axis=1) * (1 / num_edge)
            if average[current_iteration - 1] < average[current_iteration - 2] or current_iteration == max_iteration:
                break
        current_iteration += 1
    return [current_iteration, kappas]
