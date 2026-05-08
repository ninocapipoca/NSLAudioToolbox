import numpy as np
from scipy import signal as sig
from audio_toolbox.utils import mathfuncs as mf
from pathlib import Path

def wav2aud(x, paras, cochba_file=None):
    """
    Compute a fast auditory spectrogram for an acoustic waveform (band 180-7246 Hz).

    Implements a biologically-motivated auditory model using an IIR cochlear
    filterbank. Three stages are applied across ``M-1`` frequency channels,
    from highest to lowest:

    1. **Analysis** — IIR cochlear filterbank (one filter per channel)
    2. **Transduction** — hair cell nonlinearity via :func:`sigmoid`, followed
       by a low-pass hair cell membrane filter (added v2.00)
    3. **Reduction** — lateral inhibitory network (LIN), half-wave
       rectification, and temporal integration

    The filterbank spans 128 channels at 24 channels/octave, with
    characteristic frequencies defined by:

    .. math::

        CF = 440 \\times 2^{(k-31)/24}, \\quad k = 0, 1, \\ldots, 127

    giving roughly CF(60) = 1 kHz (0.5 kHz) for a 16 kHz (8 kHz) sampling rate.

    :param x: Input acoustic waveform (1-D).
    :type x: numpy.ndarray
    :param paras: Four-element parameter vector ``[frmlen, tc, fac, shft]``:

        - ``frmlen`` — frame length in ms; typical values are 8, 16, or
          powers of 2.
        - ``tc`` — leaky integration time constant in ms; typical values are
          4, 16, or 64 ms. Set to ``0`` to use short-term averaging instead
          of leaky integration.
        - ``fac`` — nonlinear factor (critical level ratio); controls the
          degree of compression applied by :func:`sigmoid`. Typical value is
          ``0.1`` for a unit-variance signal. Smaller values give more
          compression. Special values:

          - ``fac > 0`` — smooth transistor-like compression
          - ``fac = 0`` — full compression / Boolean (hard limiter)
          - ``fac = -1`` — half-wave rectifier
          - ``fac = -2`` — linear passthrough (bypasses hair cell membrane
            low-pass filter)

        - ``shft`` — octave shift relative to 16 kHz; e.g. ``0`` for 16 kHz,
          ``-1`` for 8 kHz. Sampling rate is ``SF = 16000 * 2^shft`` Hz.

    :type paras: array-like of length 4
    :param COCHBA_filename: Path to the ``.npz`` file containing the
        pre-computed IIR cochlear filterbank coefficients. Expected keys are
        ``len`` (number of channels ``M``) and, for each channel index
        ``ch``: ``zeros_<ch>``, ``poles_<ch>``, ``gain_<ch>``.
    :type COCHBA_filename: str or path-like
    :returns: Auditory spectrogram of shape ``(N, M-1)``, where ``N`` is the
        number of frames and ``M-1`` is the number of frequency channels,
        spanning 180-7246 Hz.
    :rtype: numpy.ndarray

    .. note::
        Two ``print`` statements are present in this function for debugging
        purposes. These should be replaced with :mod:`logging` calls before
        production use.

    .. rubric:: IIR Filterbank Timing Reference

    For a standard 16 kHz signal (``shft = 0``):

    +------------------+------------------+-----------+-----------+
    | Frequency range  | Downsample factor| tc (ms)   | Frame (ms)|
    +==================+==================+===========+===========+
    | 180 - 7246 Hz    | 1 / oct shift 0  | 64        | 16        |
    +------------------+------------------+-----------+-----------+
    | 90  - 3623 Hz    | 2 / oct shift -1 | 512       | 128       |
    +------------------+------------------+-----------+-----------+

    .. rubric:: References

    | Original Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
    | v1.00: 01-Jun-97
    | v1.10: 04-Sep-98 — Taishih Chi: added Kuansan's FIR filter option
    | v2.00: 24-Jul-01 — Taishih Chi: added hair cell membrane low-pass filter
    | v2.10: 04-Apr-04 — Taishih Chi: removed FIR option (see ``wav2aud_fir``)

    .. seealso::
        :func:`sigmoid`, :func:`aud2cor`

    Example usage::

        import numpy as np
        x = np.random.randn(16000)    # 1 s of white noise at 16 kHz
        paras = [8, 8, 0.1, 0]       # 8 ms frames, 8 ms tc, mild compression
        v5 = wav2aud(x, paras, 'cochba.npz')
        print(v5.shape)               # (N, M-1)
    """
    if not cochba_file:
        # if none specified, set to provided filters
        cochba_file = Path(__file__).parent / 'utils' / 'cochba_filters.npz'

    COCHBA = np.load(cochba_file, allow_pickle=True)
    M = int(COCHBA['len'])

    L_x = len(x)

    shft = paras[3]                                    # octave shift
    fac  = paras[2]                                    # nonlinear factor
    L_frm = int(np.round(paras[0] * 2**(4+shft)))     # frame length (samples)

    alph = 0
    if paras[1]:
        alph = np.exp(-1 / (paras[1] * 2**(4+shft)))  # decay factor for leaky integration

    haircell_tc = 0.5                                  # hair cell time constant (ms)
    beta = np.exp(-1 / (haircell_tc * 2**(4+shft)))   # hair cell membrane decay
    print(haircell_tc, beta)

    # Allocate output: N frames x (M-1) channels
    N = int(np.ceil(L_x / L_frm))
    x = np.pad(x, (0, int(N * L_frm - L_x)))          # zero-pad to multiple of L_frm
    x = x.reshape(-1)
    v5 = np.zeros((N, M-1))

    # --- Highest-frequency channel (channel M-1) ---
    # Process separately: no lateral inhibition at the top of the filterbank.
    z  = COCHBA[f'zeros_{M-1}']
    po = COCHBA[f'poles_{M-1}']
    k  = COCHBA[f'gain_{M-1}']

    sos = sig.zpk2sos(z, po, k)
    y1  = sig.sosfilt(sos, x).squeeze()
    y2  = mf.sigmoid(y1, fac)

    # Hair cell membrane low-pass filter (cutoff <= 4 kHz).
    # Skipped when fac == -2 (linear ionic channel mode).
    if fac != -2:
        sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
        y2 = sig.sosfilt(sos_y2, y2).squeeze()

    y2_h = y2  # store as reference for lateral inhibition in next channel

    # --- Remaining channels (high -> low frequency) ---
    for ch in range(M-2, 0, -1):
        print(f"processing channel {ch}")

        # Stage 1 — Analysis: cochlear filterbank (IIR)
        z  = COCHBA[f'zeros_{ch}']
        po = COCHBA[f'poles_{ch}']
        k  = COCHBA[f'gain_{ch}']

        sos = sig.zpk2sos(z, po, k)
        y1  = sig.sosfilt(sos, x).squeeze()

        # Stage 2 — Transduction: hair cell nonlinearity
        y2 = mf.sigmoid(y1, fac)

        # Hair cell membrane low-pass filter (cutoff <= 4 kHz).
        # Skipped when fac == -2 (linear ionic channel mode).
        if fac != -2:
            sos_y2 = sig.tf2sos(np.array([1.]), np.array([1, -beta]))
            y2 = sig.sosfilt(sos_y2, y2).squeeze()

        # Stage 3 — Reduction: lateral inhibitory network
        y3   = y2 - y2_h   # subtract higher-frequency channel response
        y2_h = y2           # update reference for next iteration
        y4   = np.maximum(y3, 0)  # half-wave rectify

        # Temporal integration
        if alph:
            # Leaky integration
            sos_y5 = sig.tf2sos(np.array([1.]), np.array([1, -alph]))
            y5 = sig.sosfilt(sos_y5, y4).squeeze()
            v5[:, ch] = sig.decimate(y5, L_frm)   # downsample with anti-aliasing
        else:
            # Short-term average over each frame
            if L_frm == 1:
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(np.reshape(y4, (N, L_frm)).T, axis=0)

    return v5

if __name__ == "__main__":
    pass