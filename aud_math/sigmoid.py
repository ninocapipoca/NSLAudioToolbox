import numpy as np

def sigmoid(y, fac):
    """
    Nonlinear function for cochlear model
    y = sigmoid(y, fac):
    fac : nonlinear factor
    -- fac > 0, transister-like function
    -- fac = 0, hard-limiter
    -- fac = -1, half-wave rectifier
    -- else, no operation (linear)

    SIGMOID is a monotonic increasing function which stimulates hair cell
    nonlinearity.
    See also: WAV2AUD, AUD2WAV

    Original Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
    v1.00: 01-Jun-97
    """
    if fac > 0:
        return 1/(1 + np.exp(-y / fac))
    elif fac == 0:
        return (y > 0).astype('float')
    elif fac == -1:
        return np.maximum(y,0)
    elif fac == -3:
        print("Not implemented yet")
        return y
    else:
        return y