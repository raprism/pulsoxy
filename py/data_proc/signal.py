"""
Note: contains parts of historic datafuncs.py (03.2006, v 0.2.2)

signal processing

"""
__author__ = 'Ralf Ahlbrink'
__date__ = '29.01.2018'
__version__ = '0.4'

import numpy as np
import pandas as pd

# dtype strings
Float64 = 'd'


def distmat(a, b):
    """
    returns the distance matrix of row vectors
    or (parametric) arrays a and b
    """

    def _error():
        raise ValueError('give 2D arrays (shape(arr)[1] must be existent)')

    try:
        xa, ya = np.shape(a)
        xb, yb = np.shape(b)
    except ValueError:
        _error()
    if ya == yb:
        y = ya
    else:
        _error()
    a = np.swapaxes(np.resize(a, (xb, xb, y)), 0, 1)
    b = np.resize(b, (xa, xa, y))
    return np.sqrt(sum((a - b)**2, 2))


def filtfilt(b, a, x, axis=-1):
    """
    filtfilt.m translation

    TODO: works only for vectors => use axis, or 2D: loop over columns
    """
    # TODO: use scipy.signal.filtfilt (but that one returns only y!)
    # TEMP
    if axis != -1:
        return

    from numpy import asarray, bmat, eye, flipud, transpose, zeros
    from scipy.signal import lfilter  #, lfiltic
    from scipy.linalg import solve

    nb = len(b)
    na = len(a)
    nfilt = max(nb, na)
    nfact = 3 * (nfilt - 1)
    xlen = len(x)

    if xlen <= nfact:
        # TODO: error/warning
        return

    # fill with zeroes to nfilt length
    b = asarray(bmat([b, zeros(nfilt - nb)]))[0]
    a = asarray(bmat([a, zeros(nfilt - na)]))[0]

    # TODO: use sparse
    A = eye(nfilt-1) \
        - bmat([transpose([-a[1:nfilt],]),
                bmat([[eye(nfilt-2)],
                      [zeros((1,nfilt-2))]])
                ])

    rhs = transpose([
        b[1:nfilt],
    ]) - transpose([
        a[1:nfilt],
    ]) * b[0]

    # initial condition
    zi = solve(A, rhs)[:, 0]

    y = asarray(
        bmat([[2 * x[0] - x[nfact + 1:1:-1]], [x],
              [2 * x[-1] - x[-2:-1 - nfact:-1]]]))[0]

    y, zii = lfilter(b, a, y, axis=axis, zi=zi * y[0])
    y, ziii = lfilter(b, a, flipud(y), axis=axis, zi=zii)
    y = flipud(y)

    return y[nfact:-nfact + 1], ziii


# moving average
def mov_avg(arr,
            wlen,
            axis=-1,
            init=np.zeros(0, dtype=Float64),
            use_filtfilt=True):
    """
    give array arr and window length

    returns the moving average (of given or last axis) of arr
    """
    from scipy.signal import lfilter, lfiltic  #, filtfilt
    alen = np.shape(arr)[axis]
    if wlen > alen:
        print('window longer than axis')
        return None
    arr = np.asarray(arr, dtype=Float64)
    b = np.ones(wlen, dtype=Float64) / float(wlen)
    a = np.ones(1, dtype=Float64)
    if not use_filtfilt and not any(init):
        return lfilter(b, a, arr, axis=axis)
    else:
        if use_filtfilt:
            ynew, xi = filtfilt(b, a, arr)
        else:
            # pb with >=2 dimensions in lfiltic, workaround for 2D
            # tested only with axes=-1
            filtic = lfiltic(b, a, y=None, x=init)
            filtic = np.array(np.size(arr) / np.shape(arr)[axis] * [filtic])
            ynew, xi = lfilter(b, a, arr, zi=filtic)
        return (ynew, xi)


def mov_avg2(arr,
             wlen,
             axis=-1,
             init=np.zeros(0, dtype=Float64),
             end=np.zeros(0, dtype=Float64),
             all=False):
    ##     if init.any(): ## and end.any():
    if np.any(init):
        init = np.asarray(init, dtype=Float64)
        end = np.asarray(end, dtype=Float64)
        ynew, xi = mov_avg(
            np.concatenate([arr, end], axis=axis),
            wlen,
            axis=axis,
            init=init,
            end=end)
        if all:
            return ynew, xi
        else:
            return (np.take(
                ynew,
                np.arange(wlen // 2, np.shape(ynew)[axis] - wlen // 2),
                axis=axis), xi)
    else:
        return (mov_avg(arr, wlen, axis), np.zeros(0, 'd'))


def contourc(arr, levels=None, level=None):
    # similar to Matlab's contourc(Z, V)
    from matplotlib._contour import Cntr
    from matplotlib.mlab import meshgrid
    xlen, ylen = np.shape(arr)
    xC, yC = meshgrid(list(range(xlen)), list(range(ylen)))
    Cc = Cntr(xC, yC, arr.T)
    if np.asarray(levels).any():
        return Cc.trace(*levels)
    elif level:
        return Cc.trace(level)
    else:
        return []


def savitzky_golay(data, kernel=11, order=4):
    """
        applies a Savitzky-Golay filter
        input parameters:
        - data => data as a 1D numpy array
        - kernel => a positiv integer > 2*order giving the kernel size
        - order => order of the polynomal
        returns smoothed data as a numpy array

        invoke like:
        smoothed = savitzky_golay(<rough>, [kernel = value], [order = value]
    """
    # from www.scipy.org/Cookbook/SavitzkyGolay
    import numpy.linalg as linalg
    try:
        kernel = abs(int(kernel))
        order = abs(int(order))
    except ValueError as msg:
        raise ValueError(
            "kernel and order have to be of type int (floats will be converted)."
        )
    if kernel % 2 != 1 or kernel < 1:
        raise TypeError(
            "kernel size must be a positive odd number, was: %d" % kernel)
    if kernel < order + 2:
        raise TypeError(
            "kernel is to small for the polynomals\nshould be > order + 2")

    # a second order polynomal has 3 coefficients
    order_range = list(range(order + 1))
    half_window = (kernel - 1) // 2
    b = np.mat([[k**i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    # since we don't want the derivative, else choose [1] or [2], respectively
    m = linalg.pinv(b).A[0]
    window_size = len(m)
    half_window = (window_size - 1) // 2

    # precompute the offset values for better performance
    offsets = list(range(-half_window, half_window + 1))
    offset_data = list(zip(offsets, m))

    smooth_data = list()

    # temporary data, with padded zeros (since we want the same length after smoothing)
    data = np.concatenate((np.zeros(half_window), data, np.zeros(half_window)))
    for i in range(half_window, len(data) - half_window):
        value = 0.0
        for offset, weight in offset_data:
            value += weight * data[i + offset]
        smooth_data.append(value)
    return np.array(smooth_data)


def splp_wrapper(df, columns=['x', 'y'], k=3, s=1, nfact=1, debug=False):
    from scipy.interpolate import splprep, splev
    xy = df[columns].T.values.tolist()
    u = np.arange(len(df))
    step = 1. / nfact
    u_fine = np.arange(u[0], u[-1] + step, step)
    tck_u, fp, ier, msg = splprep(x=xy, u=u, k=k, s=s, full_output=True)
    x_sp, y_sp = splev(u_fine, tck_u[0])
    ret = pd.DataFrame(np.c_[x_sp, y_sp], columns=columns)
    if debug:
        return ret, [fp, ier, msg]
    else:
        return ret


def detect_peaks(x,
                 mph=None,
                 mpd=1,
                 threshold=0,
                 edge='rising',
                 kpsh=False,
                 valley=False,
                 show=False,
                 ax=None):
    """Detect peaks in data based on their amplitude and other features.
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) >
                                                        0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >=
                                                       0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(
            ind,
            np.unique(np.hstack((indnan, indnan - 1, indnan + 1))),
            invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(
            np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        detect_peaks_plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def get_peaks(df, colname, *args, **kw_args):
    _ = kw_args.pop('show', None)
    idx = detect_peaks(df[colname], *args, **kw_args)
    return df.iloc[idx][colname]


def detect_peaks_plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(
                ind,
                x[ind],
                '+',
                mfc=None,
                mec='r',
                mew=2,
                ms=8,
                label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')" %
                     (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def affine_regression(x_values, y_values):
    """
    Source: Python Papers Source Codes . 2010, Vol. 2, p1-7. 7p.
    Author(s): Kloss, Guy K.; Kloss, Tim F.
    Subject Terms: *REGRESSION analysis   *VECTOR fields   *AFFINE geometry   *LEAST squares   *APPROXIMATION theory   *DIMENSION reduction (Statistics)

    This function solves the linear equation system involved in the n
    dimensional linear extrapolation of a vector field to an arbitrary point.
    f(x) = x * A + b
    with:
    A - The "slope" of the affine function in an n x n matrix.
    b - The "offset" value for the n dimensional zero vector.
    The function takes a list of n+1 point-value tuples (x, f(x)) and returns
    the matrix A and the vector b. In case anything goes wrong, the function
    returns the tuple (None, None).
    These can then be used to compute directly any value in the linear
    vector field.

    # local
    RA: small modification to distinct x and y inputs
    """
    # Some helpers.
    dimensions = len(x_values[0])
    unknowns = dimensions**2 + dimensions
    number_points = len(x_values)
    # Bail out if we do not have enough data.
    if number_points < unknowns:
        print('For a {} dimensional problem I need at least {} data points.'.
              format(dimensions, unknowns))
        print('Only {} data points were given.'.format(number_points))
        return None, None

    # For the solver we are stating the problem as
    # C * x = d
    # with the problem_matrix C and the problem_vector d

    # We're going to feed our linear problem into these arrays.
    # This one is the matrix C.
    problem_matrix = np.zeros([unknowns, unknowns])
    # This one is the vector d.
    problem_vector = np.zeros([unknowns])

    # Populate data matrix C and vector d.
    x_values, y_values = np.asarray(x_values), np.asarray(y_values)
    for i in range(dimensions):
        x_i, y_i = x_values[:, i], y_values[:, i]
        for j in range(dimensions):
            y_j = y_values[:, j]
            row = dimensions * i + j
            problem_vector[row] = (x_i * y_j).sum()
            problem_matrix[row, dimensions**2 + j] = x_i.sum()
            problem_matrix[dimensions**2 + j, dimensions * i + j] = x_i.sum()
            for k in range(dimensions):
                x_k = x_values[:, k]
                problem_matrix[row, dimensions * k + j] = (x_i * x_k).sum()
        row = dimensions**2 + i
        problem_vector[row] = y_i.sum()
        problem_matrix[row, dimensions**2 + i] = number_points

    matrix_A, vector_b = None, None
    try:
        result_vector = np.linalg.solve(problem_matrix, problem_vector)
        # Check whether we really did get the right answer.
        # This is advised by the NumPy doc string.
        if np.linalg.norm(
                np.dot(problem_matrix, result_vector) - problem_vector) < 1e-6:
            # We're good, so hack up the result into the matrix and vector.
            matrix_A = result_vector[:dimensions**2]
            matrix_A.shape = (dimensions, dimensions)
            vector_b = result_vector[dimensions**2:]
        else:
            print("For whatever reason our linear equations didn't solve.")
            print(
                np.linalg.norm(
                    np.dot(result_vector, problem_matrix) - problem_vector))
    except np.linalg.linalg.LinAlgError:
        print("Things didn't work out as expected, eh.")
    return matrix_A, vector_b


def get_clusters_single1D(pos, d_thres):
    """returns distance threshold (d_thres)
       based cluster of pos vector
    """
    # - simpler and faster alternative
    #   to vector-only usage of distance-based fastcluster
    # - in comparison contains also clusters consist of single points
    pos = np.atleast_1d(np.squeeze(pos))
    assert pos.ndim == 1, '## pos input is not a vector'
    if len(pos):
        cl_first = [pos[0]]
        cl_last = []
        for i in range(len(pos) - 1):
            if pos[i + 1] - pos[i] >= d_thres:
                cl_last.append(pos[i])
                cl_first.append(pos[i + 1])
        cl_last.append(pos[-1])
    else:
        cl_first = cl_last = []
    return np.c_[cl_first, cl_last]


def get_regions(var, clusters, var2=None):
    """searches values in var as defined in clusters (list of [beg, end] entries)
       and returns var-sized float array with cluster regions, and gaps set as NaNs
    """
    prepf = lambda v: np.squeeze(np.asarray(v))
    var = prepf(var)
    clusters = clusters.astype(var.dtype)
    regions = np.nan * np.ones(var.shape, dtype=np.float)
    if var2 is not None:
        assert var2.shape == var.shape, '## incompatible shape of '
        var2 = prepf(var2)
        regions2 = regions.copy()
    acond = np.zeros(var.shape, dtype=np.bool)
    sorter = np.arange(len(var))
    for cl in clusters:
        beg, end = np.searchsorted(var, cl, sorter=sorter)
        acond[beg:end] = True
    regions[acond] = var[acond]
    res = [regions]
    if var2 is not None:
        regions2[acond] = var2[acond]
        res.append(regions2)
    return res


## matlab like plotting mfreqz and impz
## (http://mpastell.com/2010/01/18/fir-with-scipy/)

#Plot frequency and phase response
def mfreqz(b, a=1, fig_num=None):
    from scipy import signal
    import matplotlib.pyplot as plt
    w, h = signal.freqz(b, a)
    h_dB = 20 * np.log10(abs(h))
    fig, axs = plt.subplots(2, 1, sharex=True, num=fig_num)
    ax = axs[0]
    ax.plot(w / max(w), h_dB)
    ax.set_ylim(-150, 5)
    ax.set_ylabel('Magnitude (db)')
    ax.set_title(r'Frequency response')
    ax = axs[1]
    h_Phase = np.unwrap(np.angle(h))
    ax.plot(w / max(w), h_Phase)
    ax.set_ylabel('Phase (radians)')
    ax.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    ax.set_title(r'Phase response')
    #fig.subplots_adjust(hspace=0.5)
    return fig


#Plot step and impulse response
def impz(b, a=1, fig_num=None):
    from scipy import signal
    import matplotlib.pyplot as plt
    l = len(b)
    impulse = np.repeat(0., l)
    impulse[0] = 1.
    x = np.arange(0, l)
    response = signal.lfilter(b, a, impulse)
    fig, axs = plt.subplots(2, 1, sharex=True, num=fig_num)
    ax = axs[0]
    ax.stem(x, response)
    ax.set_ylabel('Amplitude')
    ax.set_title(r'Impulse response')
    ax = axs[1]
    step = np.cumsum(response)
    ax.stem(x, step)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel(r'n (samples)')
    ax.set_title(r'Step response')
    #fig.subplots_adjust(hspace=0.5)
    return fig
