import numpy as np

def sigmoid(y, fac):
    """
    Nonlinear sigmoid function for cochlear hair cell modelling.

    A monotonic increasing function that simulates hair cell nonlinearity.
    Behaviour is controlled by the ``fac`` parameter, which selects between
    several operating modes ranging from a smooth transistor-like response
    to a hard limiter or half-wave rectifier.

    :param y: Input signal array.
    :type y: numpy.ndarray
    :param fac: Nonlinear factor controlling the operating mode:

        - ``fac > 0`` — transistor-like sigmoid (smooth nonlinearity)
        - ``fac = 0`` — hard limiter (Heaviside step function)
        - ``fac = -1`` — half-wave rectifier (clips negative values to zero)
        - ``fac = -3`` — reserved (not yet implemented, returns ``y`` unchanged)
        - otherwise — linear passthrough (no operation)

    :type fac: float
    :returns: Transformed signal, same shape as ``y``.
    :rtype: numpy.ndarray

    .. note::
        When ``fac = -3``, a warning is printed and the input is returned
        unchanged. This mode is reserved for future implementation.

    .. seealso::
        :func:`wav2aud`

    .. rubric:: References

    | Original Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
    | v1.00: 01-Jun-97

    Example usage::

        import numpy as np
        y = np.linspace(-5, 5, 100)

        # Smooth sigmoid
        out = sigmoid(y, fac=0.1)

        # Hard limiter
        out = sigmoid(y, fac=0)

        # Half-wave rectifier
        out = sigmoid(y, fac=-1)
    """

    if fac > 0:
        return 1/(1 + np.exp(-y / fac))
    elif fac == 0:
        return (y > 0).astype('float')
    elif fac == -1:
        return np.maximum(y, 0)
    elif fac == -3:
        print("Not implemented")
        return y
    else:
        return y