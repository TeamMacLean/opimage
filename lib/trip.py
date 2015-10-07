from __future__ import division

import numpy as np
import math
import os
from scipy.misc import imread, imsave, imresize
import glob
import re


def strtok(*args):
    return [str.upper() for str in args]


def fclose(file):
    file.close()


def numel(a):
    # return a.size
    return np.asarray(a).size


def safeOpenFle(file):
    try:
        fp = open(file)
        assert fp != -1
        return fp
    except:
        return -1


def uniqueImages():
    dirName = '../input'
    acceptedType = [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ds = []
    for file in os.listdir(dirName):
        if file.endswith(tuple(acceptedType)):
            ds.append(file)

    d = np.array(ds)
    d = np.unique(d)
    return d


def cropAll(fn=None):
    fd = safeOpenFle(fn)
    if fd == - 1:
        print 'File %s does not exist... aborting' % fn
        return
    fclose(fd)

    dirName = '../input'
    d = uniqueImages()
    print "Found %d images." % numel(d)

    # return

    for k in np.arange(0, numel(d)).reshape(-1):
        # print"processing %d/%d..." % (k, numel(d))
        im = imread("%s/%s" % (dirName, d[k]))
        print "reading %s" % d[k]
        fd = safeOpenFle(fn)
        doneAllLines = 0

        # for each line crop and save an image
        while doneAllLines == 0:
            line = fd.readline().rstrip()
            if len(line) > 0:
                if line == - 1:
                    fclose(fd)
                    break
                subdir, y1, y2, x1, x2 = re.split(' ', line)
                subdir = "%s/crop_%s" % (dirName, subdir)
                y1 = int(y1)
                y2 = int(y2)
                x1 = int(x1)
                x2 = int(x2)

                if os.path.isfile("%s/crop_%s" % (subdir, d[k])):
                    print "skipping %s/crop_%s." % (subdir, d[k])
                else:
                    if k == 0 and not os.path.exists(subdir):
                        os.makedirs(subdir)

                    imC = im[y1:y2, x1:x2, :]
                    name = "%s/crop_%s" % (subdir, d[k])
                    imsave(name, imC)

            else:
                doneAllLines = 1

    return


# cropAll('./testdata.txt')

def size(mat, elem=None):
    if not elem:
        return mat.shape
    else:
        return mat.shape[int(elem) - 1]


def space_time_deriv(f=None):
    f = np.array(f)
    N = f.shape[0]
    # print N
    deriv = []
    pre = []
    dims = f[0].im.shape

    # print N
    # print dims

    if N == 2.:
        pre = np.array(np.hstack((0.5, 0.5)))
        deriv = np.array(np.hstack((-1., 1.)))
    elif N == 3.:
        pre = np.array(np.hstack((0.223755, 0.552490, 0.223755)))
        deriv = np.array(np.hstack((-0.453014, 0.0, 0.453014)))

    elif N == 4.:
        pre = np.array(np.hstack((0.092645, 0.407355, 0.407355, 0.092645)))
        deriv = np.array(np.hstack((-0.236506, -0.267576, 0.267576, 0.236506)))

    elif N == 5.:
        pre = np.array(np.hstack((0.036420, 0.248972, 0.429217, 0.248972, 0.036420)))
        deriv = np.array(np.hstack((-0.108415, -0.280353, 0.0, 0.280353, 0.108415)))

    elif N == 6.:
        pre = np.array(np.hstack((0.013846, 0.135816, 0.350337, 0.350337, 0.135816, 0.01384)))
        deriv = np.array(np.hstack((-0.046266, -0.203121, -0.158152, 0.158152, 0.203121, 0.046266)))

    elif N == 7.:
        pre = np.array(np.hstack((0.005165, 0.068654, 0.244794, 0.362775, 0.244794, 0.068654, 0.005165)))
        deriv = np.array(np.hstack((-0.018855, -0.123711, -0.195900, 0.0, 0.195900, 0.123711, 0.018855)))

    else:
        raise "No such filter size (N=%d)" % N

    # fdt = []
    # fpt = []
    fdt = np.zeros(dims)
    fpt = np.zeros(dims)

    for i in np.arange(0, N).reshape(-1):
        fdt = fdt + deriv[i] * f[i].im
        fpt = fpt + pre[i] * f[i].im

        # d = fdt + deriv[i] * f[i].im
        # p = fpt + pre[i] * f[i].im

    print fdt.shape
    print fpt.shape

    print fdt[0].shape
    print fpt[0].shape

    print fdt[0][0]

    # fdt = d[0] #fake fix
    # fpt = p[0] #fake fix

    # print(fdt[0])

    # print fdt.shape
    # print fpt.shape

    fx = np.convolve(np.convolve(fpt, pre.conj().T, 'same'), deriv, 'same')
    fy = np.convolve(np.convolve(fpt, pre, 'same'), deriv.conj().T, 'same')
    ft = np.convolve(np.convolve(fdt, pre.conj().T, 'same'), pre, 'same')

    # f1 = np.convolve(fpt, pre.conj().transpose(), mode='same')
    # f1 = convolve2d(fpt, pre.conj().transpose(), 'same')
    # fx = convolve2d(f1, deriv, 'same')

    # f2 = convolve2d(fpt, pre, 'same')
    # fy = convolve2d(f2, deriv.conj().T, 'same')

    # f3 = convolve2d(fdt, pre.conj().T, 'same')
    # ft = convolve2d(f3, pre, 'same')
    return fx, fy, ft


def errorFunc_(model=None, dat=None):
    N = len(dat)
    f = evaluateModel_(model, N)
    err = sum_((f - dat) ** 2)
    return err


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


class imGroup:
    def __init__(self, orig, im):
        self.orig = orig
        self.im = im


def estimateMotion(dirname=None):
    GRADIENT_THRESHOLD = 8
    DISPLAY = 0
    d = uniqueImages()
    N = numel(d)
    print N
    scale = float(1)
    print "loading %d frames..." % N
    f = []
    for k in np.arange(0, N).reshape(-1):
        im = imread("%s/crop_%s" % (dirname, d[k]))

        if k == 0:
            scale = 60. / max(size(im))
        im = imresize(im, scale, 'bicubic')
        orig = im
        im = np.double(rgb2gray(im))  # TODO MAKE SECOND GREY (and a double)
        o = imGroup(orig, im)
        f.append(o)

    # end
    ydim, xdim = f[0].im.shape[:2]

    # print "orig"
    # print f[0].orig.shape
    # print "im"
    # print f[0].im.shape

    print "computing motion..."
    taps = 7
    blur = np.array([1, 6, 15, 20, 15, 6, 1])
    blur = blur / blur.sum()
    s = 1
    N = numel(f) - taps  # -1
    Vx = np.zeros((ydim / s, xdim / s, N))
    Vy = np.zeros((ydim / s, xdim / s, N))
    f = np.array(f)

    for k in np.arange(0, N).reshape(-1):
        fx, fy, ft = space_time_deriv(f[k:k + taps])  # -1
        fx2 = np.convolve(np.convolve(fx.dot(fx), blur.T, 'same'), blur, 'same')
        fy2 = np.convolve(np.convolve(fy.dot(fy), blur.T, 'same'), blur, 'same')
        fxy = np.convolve(np.convolve(fx.dot(fy), blur.T, 'same'), blur, 'same')
        fxt = np.convolve(np.convolve(fx.dot(ft), blur.T, 'same'), blur, 'same')
        fyt = np.convolve(np.convolve(fy.dot(ft), blur.T, 'same'), blur, 'same')

        print fx.shape
        print fy.shape

        grad = np.sqrt((fx ** 2. + fy ** 2.))

        print grad.shape

        grad[:, 0:5.] = 0.
        grad[0:5., :] = 0.
        grad[:, int(xdim - 4.) - 1:xdim] = 0.
        grad[int(ydim - 4.) - 1:ydim, :] = 0.
        cx = 1
        bad = 0
        for x in np.arange(0, xdim, s).reshape(-1):
            cy = 1
            for y in np.arange(0, ydim, s).reshape(-1):
                M = np.array([[fx2[y, x], fxy[y, x]], [fxy[y, x], fy2[y, x]]])
                b = np.array([[fxt[y, x]], [fyt[y, x]]])
                if np.linalg.cond(M) > 100.0 or grad[y, x] < GRADIENT_THRESHOLD:
                    Vx[cy, cx, k] = 0
                    Vy[cy, cx, k] = 0
                    bad += 1
                else:
                    v = inv_(M) * b
                    Vx[cy, cx, k] = v[1]
                    Vy[cy, cx, k] = v[2]
                cy = cy + 1
            cx = cx + 1
        if (bad / (xdim * ydim) == 1):
            print "WARNING on frame %d: no velocity estimate" % k
    taps = 13
    blur = ones_(1, taps)
    blur = blur / sum_(blur)
    if (DISPLAY):
        figure(1)
        for k in np.arange(0, N - taps).reshape(-1):
            vx = np.zeros(size(Vx, 1), size_(Vx, 2))
            vy = np.zeros(size(Vy, 1), size_(Vy, 2))
            Vx2 = Vx[:, :, k:k + taps - 1]
            Vy2 = Vy[:, :, k:k + taps - 1]
            for j in np.arange(0, length_(blur)).reshape(-1):
                vx = vx + blur[j] * Vx2[:, :, j]
                vy = vy + blur[j] * Vy2[:, :, j]
            imagesc_(f[k + floor_(taps / 2)].orig)
            axis_(char('image'), char('off'))
            colormap_(char('gray'))
            xramp, yramp = meshgrid_([np.arange(0, xdim, s)], [np.arange(0, ydim, s)], nargout=2)
            hold_(char('on'))
            ind = find_(vx == 0 and vy == 0)
            xramp[ind] = 0
            yramp[ind] = 0
            h = quiver_(xramp, yramp, 20 * vx, 20 * vy, 0)
            set_(h, char('Color'), char('r'), char('LineWidth'), 1)
            hold_(char('off'))
            drawnow
    c = 1
    for k in np.arange(0, N - taps).reshape(-1):
        vx = zeros_(size_(Vx, 1), size_(Vx, 2))
        vy = zeros_(size_(Vy, 1), size_(Vy, 2))
        Vx2 = Vx[:, :, k:k + taps - 1]
        Vy2 = Vy[:, :, k:k + taps - 1]
        for j in np.arange(0, length_(blur)).reshape(-1):
            vx = vx + blur[j] * Vx2[:, :, j]
            vy = vy + blur[j] * Vy2[:, :, j]
        indx = find_(abs_(vx) > eps)
        indy = find_(abs_(vy) > eps)
        motion_x[c] = 1 / scale * mean_(vx[indx])
        motion_y[c] = - 1 / scale * mean_(vy[indy])
        c = c + 1
    return motion_x, motion_y


def evaluateModel_(model=None, N=None):
    freq = model[1]
    phase = model[2]
    amp = model[3]
    t = np.array([np.arange(0, N - 1)])
    f = amp.dot(cos_(freq * 2 * pi / N * t + phase))
    return f


def modelFitAll_():
    d = dir_(char('../output/*.csv'))
    for k in np.arange(0, numel(d)).reshape(-1):
        fn = np.array([char('../output/'), d[k].name])
        dat = csvread_(fn)
        dat = dat.T
        dat[isnan_(dat)] = 0
        dat = dat - mean_(dat)
        dat = detrend_(dat)
        N = length_(dat)
        D = fftshift_(fft_(dat))
        if (mod_(length_(dat), 2) == 0):
            mid = length_(dat) / 2 + 1
        else:
            mid = floor_(length_(dat) / 2) + 1
        D = D[mid:mid + 10]
        val, ind = max_(abs_(D), nargout=2)
        freq = ind - 1
        phase = angle_(D[ind])
        amp = mean_(abs_(dat))
        model = fminsearch_(char('errorFunc'), [freq, phase, amp], [], dat)
        fnout = strrep_(fn, char('.csv'), char('_model.txt'))
        fdout = fopen_(fnout, char('w'))
        fprintf_(fdout, char('%f\\n'), model[1])
        fclose_(fdout)
        fnout = strrep_(fn, char('.csv'), char('_model.png'))
        f = evaluateModel_(model, N)
        plot_(dat, char('b'))
        hold_(char('on'))
        plot_(f, char('r'))
        axis_([0, N - 1, min_(dat), max_(dat)])
        legend_(char('data'), char('model'))
        title_(sprintf_(char('frequency = %f'), model[1]))
        hold_(char('off'))
        drawnow
        FRAME = getframe_(gcf)
        imwrite_(uint8_(frame2im_(FRAME)), fnout)
    return


def estimateAll():
    indirname = '../input'
    outdirname = '../output'
    d2 = []
    d = glob.glob("%s/crop_*" % indirname)
    for k in np.arange(0, numel(d)).reshape(-1):  # TODO 0 or 1
        if os.path.isdir(d[k]) == 1:
            d2.append(d[k])
    d = np.array(d2).copy()
    print "found %d directories." % numel(d)
    for k in np.arange(0, numel(d)).reshape(-1):  # TODO 0 or 1
        if os.path.isfile("../%s/%s.csv" % (outdirname, d[k])):
            print "%s already exists... skipping." % d[k]
        else:
            try:
                motion_x, motion_y = estimateMotion("%s/%s" % (indirname, d[k]))
            finally:
                pass
            fd = safeOpenFle("%s/%s.csv" % (outdirname, d[k].name))
            # for j in np.arange(0, length_(motion_y)).reshape(-1):
            #     fprintf_(fd, char('%f\\n'), motion_y[j])
            # fclose_(fd)
            # figure_(1)
            # cla
            # hold_(char('on'))
            # plot_(motion_y, char('k'))
            # hold_(char('off'))
            # legend_(char('vertical'))
            # xlabel_(char('frame'))
            # ylabel_(char('motion (pixels/frame)'))
            # title_(d[k].name)
            # box_(char('on'))
            # axis_(char('tight'))
            # FRAME = getframe_(gcf)
            # imwrite_(uint8_(frame2im_(FRAME)), sprintf_(char('%s/%s.png'), outdirname, d[k].name))
    return


# cropAll('./testdata.txt')
estimateAll()
