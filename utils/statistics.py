import cv2
import numpy as np


def is_inside(pos, contours, scale, imgsize, min_dist=200, summaryfun=min):
    # > 0 if pos in any of the contours
    pos2 = (pos[0] + imgsize * (2 ** scale),
            pos[1])
    pos3 = (pos[0],
            pos[1] + imgsize * (2 ** scale))
    pos4 = (pos[0] + imgsize * (2 ** scale),
            pos[1] + imgsize * (2 ** scale))

    maxdistance = max([cv2.pointPolygonTest(contour, pos, True) for contour in contours])
    maxdistance2 = max([cv2.pointPolygonTest(contour, pos2, True) for contour in contours])
    maxdistance3 = max([cv2.pointPolygonTest(contour, pos3, True) for contour in contours])
    maxdistance4 = max([cv2.pointPolygonTest(contour, pos4, True) for contour in contours])
    mindistance = summaryfun(maxdistance, maxdistance2, maxdistance3, maxdistance4)

    return mindistance > min_dist, "{0:04d}".format(int(mindistance / 100))


def blur(img, channel=0):
    gray = img[:, :, channel]
    b = int(blur_fft(gray))
    return "{0:04d}".format(b)


def blur_fft(image, size=60):
    h, w = image.shape
    cX, cY = int(w / 2.0), int(h / 2.0)

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    blur_mag = np.mean(20 * np.log(np.abs(recon)))
    if np.isinf(blur_mag):
        blur_mag = -1000000
    if blur_mag < 0:
        blur_mag = 0
    return blur_mag * 10.0


def positive_rate(ihc, th):
    # return permil of positive value
    all_pixel = ihc.size

    positive_pixel = ihc[np.where(ihc > th)].size

    permil2 = int(10000.0 * float(positive_pixel) / float(all_pixel))

    return "{0:05d}".format(permil2)


def threshold_otsu2(hist):
    bin_centers = np.arange(256)
    hist = hist.astype(float)

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold
