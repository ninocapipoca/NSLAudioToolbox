import numpy as np
from scipy import signal as sig
from ..aud_math.sigmoid import sigmoid

def wav2aud(x, paras, COCHBA_filename):
    """
    Helpful docstring goes here
    """
    COCHBA = np.load(COCHBA_filename, allow_pickle=True)
    M = int(COCHBA['len'])

    L_x = len(x) # length of input x

    shft = paras[3] # octave shift
    fac = paras[2] # nonlinear factor
    L_frm = int(np.round(paras[0] * 2**(4+shft))) # frame length (points)

    alph = 0 # decay factor
    if paras[1]: # if second arg nonzero
        alph = np.exp(-1 / (paras[1] * 2**(4+shft)))

    haircell_tc = 0.5; # hair cell time constant, ms
    beta = np.exp(-1 / (haircell_tc * 2**(4+shft)))
    print(haircell_tc, beta)

    # allocating memory for output
    N = int(np.ceil(L_x / L_frm))  # num frames

    x = np.pad(x, (0, int(N * L_frm - L_x))) # zero-padding
    x = x.reshape(-1) # convert to 1D array; column vect
    v4, v5 = np.zeros((N, M-1)), np.zeros((N, M-1))

    ########################################
    # last channel (highest frequency)
    ########################################

    z = COCHBA[f'zeros_{M-1}']
    po = COCHBA[f'poles_{M-1}']
    k = COCHBA[f'gain_{M-1}']

    sos = sig.zpk2sos(z, po, k)
    y1 = sig.sosfilt(sos, x).squeeze()
    print("y1 ------\n", y1[:10])
    y2 = sigmoid(y1, fac)

    # hair cell membrane (low pass <= 4 kHz)
    # ignored for linear ionic channels
    if fac != -2:
        sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
        y2 = sig.sosfilt(sos_y2, y2).squeeze()
    
    y2_h = y2

    ########################################
    # all other channels
    ########################################
    for ch in range(M-2, 0, -1):# was (M-1, 0, -1) (M-2, 50, -1):
        print(f"processing channel {ch}")
        ########################################
        # analysis: cochlear filterbank
        ########################################
        # IIR filterbank convolution ----> y1

        z = COCHBA[f'zeros_{ch}']
        po = COCHBA[f'poles_{ch}']
        k = COCHBA[f'gain_{ch}']

        sos = sig.zpk2sos(z, po, k)
        y1 = sig.sosfilt(sos, x).squeeze()
        ########################################
        # transduction: hair cells
        ########################################

        # ionic channels (sigmoid function)
        y2 = sigmoid(y1, fac)

        # hair cell membrane (low-pass <= 4 kHz) ---> y2 (ignored for linear)
        if fac != -2:
            sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
            y2 = sig.sosfilt(sos, y2).squeeze()

        ########################################
        # reduction: lateral inhibitory network
        ########################################
        # masked by higher (frequency) spatial response
        y3 = y2 - y2_h
        y2_h = y2

        # half-wave rectifier ----> y4
        y4 = np.maximum(y3, 0)

        # temporal integration window
        if alph: # leaky integration
            sos_y5 = sig.tf2sos(np.array([1.]), np.array([1, -alph]))
            y5 = sig.sosfilt(sos_y5, y4).squeeze()
            v5[:, ch] = sig.decimate(y5, L_frm) # anti aliasing filter
        else: # short term average
            if L_frm == 1:
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(np.reshape(y4, (N, L_frm)).T, axis=0)

    return v5

