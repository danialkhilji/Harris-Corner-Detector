import cv2
import numpy as np
from skimage.color import rgb2gray
import timeit
import matplotlib.pyplot as plt


def main():
    start = timeit.default_timer()
    im = cv2.imread("image2.jpg")
    img = rgb2gray(im)
    cv2.imshow("Image in Gray", img)
    # cv2.waitKey(0)
    (h, w) = img.shape[:2]
    print("Height: {}, Widht: {} ".format(h, w))

    # Part A
    print("Part A")
    cnr = corner(img)
    dst = cv2.addWeighted(img, 1, cnr, 0.5, 0)
    cv2.imshow("Edges", dst)
    # cv2.waitKey(0)

    # Part B
    print("Part B")
    rotation(img)

    # Part C
    print("Part C")
    scaling(img)

    end = timeit.default_timer()
    print("Run time", end - start)


def corner(img):
    # X-axis
    img1 = img.copy()
    ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    cv2.imshow("SobelX", ix)
    # cv2.waitKey(0)

    # Y-axis
    img2 = img.copy()
    iy = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)
    cv2.imshow("SobelY", iy)
    # cv2.waitKey(0)

    ixsq = cv2.multiply(ix, ix)
    ix2 = cv2.GaussianBlur(ixsq, (3, 3), 0)
    iysq = cv2.multiply(iy, iy)
    iy2 = cv2.GaussianBlur(iysq, (3, 3), 0)
    ixiy = cv2.multiply(ix, iy)
    ixy = cv2.GaussianBlur(ixiy, (3, 3), 0)

    # Determinant
    a = cv2.multiply(ix2, iy2)
    b = cv2.multiply(ixy, ixy)
    det = cv2.subtract(a, b)

    # Trace
    trc = cv2.add(ix2, iy2)
    trc2 = 0.04 * cv2.multiply(trc, trc)

    # Corner score
    cnr = cv2.subtract(det, trc2)
    cv2.imshow("Corner score", cnr)
    # cv2.waitKey(0)
    ret, thresh = cv2.threshold(cnr, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh", thresh)
    # cv2.waitKey(0)
    return thresh


def rotation(img):
    (h, w) = img.shape[:2]
    cntr = (w/2, h/2)
    cnr = corner(img)
    rtn = 15
    M = []
    degree = []

    for i in range(24):
        print("At {} degree rotation: ".format(rtn))
        # rotation of corners
        orig_mat = cv2.getRotationMatrix2D(cntr, rtn, 1)
        cnr15 = cv2.warpAffine(cnr, orig_mat, (w, h))
        # rotation of grayscale image
        rtn_mat = cv2.getRotationMatrix2D(cntr, rtn, 1)
        rtn15 = cv2.warpAffine(img, rtn_mat, (w, h))
        cv2.imshow("Rotation", rtn15)
        # cv2.waitKey(0)

        # corners after rotation
        cnrtn15 = corner(rtn15)
        cv2.imshow("Corners after rotation", cnrtn15)
        # cv2.waitKey(0)
        degree.append(rtn)
        rtn += 15

        (r_height, r_width) = img.shape[:2]
        blank = np.zeros((r_height, r_width), np.uint8)
        match = 0
        for u in range(1, 477):
            for v in range(1, 637):
                if cnr15[u][v] != 0 and cnrtn15[u][v] != 0:
                    cnrtn15[u][v] = 0
                    blank[u][v] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u - 1][v] != 0:
                    cnrtn15[u - 1][v] = 0
                    blank[u - 1][v] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u - 1][v - 1] != 0:
                    cnrtn15[u - 1][v - 1] = 0
                    blank[u - 1][v - 1] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u][v - 1] != 0:
                    cnrtn15[u][v - 1] = 0
                    blank[u][v - 1] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u + 1][v] != 0:
                    cnrtn15[u + 1][v] = 0
                    blank[u + 1][v] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u + 1][v + 1] != 0:
                    cnrtn15[u + 1][v + 1] = 0
                    blank[u + 1][v + 1] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u - 1][v + 1] != 0:
                    cnrtn15[u - 1][v + 1] = 0
                    blank[u - 1][v + 1] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u][v + 1] != 0:
                    cnrtn15[u][v + 1] = 0
                    blank[u][v + 1] = 255
                    match = match + 1
                elif cnr15[u][v] != 0 and cnrtn15[u + 1][v - 1] != 0:
                    cnrtn15[u + 1][v - 1] = 0
                    blank[u + 1][v - 1] = 255
                    match = match + 1
        M.append(match)
        print("Matches: ", match)
    N = key_points_n(img)
    repeatability = []
    for i in range(len(M)):
        repeatability.append(M[i]/N)
    print("Repeatability: ", repeatability)

    plt.figure(1)
    plt.plot(degree, repeatability, color="blue", marker=".")
    plt.xticks(degree, rotation=90)
    plt.yticks(repeatability)
    plt.xlabel("Degree rotation")
    plt.ylabel("Match frequency")
    plt.title("Rotation graph")
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.waitforbuttonpress()


def scaling(img):
    # corners of original image
    cnr_orig = corner(img)
    m = 1.2
    M = []
    fct = []
    for i in range(8):
        factor = m**i
        print("Scaling factor:", factor)
        # scaling of corners (cv2.INTER_CUBIC is for bicubic interpolation)
        cnr_scld = cv2.resize(cnr_orig, None, None, factor, factor, cv2.INTER_CUBIC)
        # scaling of original image
        scld_img = cv2.resize(img, None, None, factor, factor, cv2.INTER_CUBIC)
        cv2.imshow("Scaled image", scld_img)
        # cv2.waitKey(0)
        # corners of scaled image
        cnr_img = corner(scld_img)
        cv2.imshow("Corners after scaling", cnr_img)
        # cv2.waitKey(0)

        (s_height, s_width) = cnr_img.shape[:2]
        blank = np.zeros((s_height, s_width), np.uint8)
        match = 0
        for u in range(1, 477):
            for v in range(1, 637):
                if cnr_scld[u][v] != 0 and cnr_img[u][v] != 0:
                    cnr_img[u][v] = 0
                    blank[u][v] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u - 1][v] != 0:
                    cnr_img[u - 1][v] = 0
                    blank[u - 1][v] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u - 1][v - 1] != 0:
                    cnr_img[u - 1][v - 1] = 0
                    blank[u - 1][v - 1] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u][v - 1] != 0:
                    cnr_img[u][v - 1] = 0
                    blank[u][v - 1] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u + 1][v] != 0:
                    cnr_img[u + 1][v] = 0
                    blank[u + 1][v] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u + 1][v + 1] != 0:
                    cnr_img[u + 1][v + 1] = 0
                    blank[u + 1][v + 1] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u - 1][v + 1] != 0:
                    cnr_img[u - 1][v + 1] = 0
                    blank[u - 1][v + 1] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u][v + 1] != 0:
                    cnr_img[u][v + 1] = 0
                    blank[u][v + 1] = 255
                    match = match + 1
                elif cnr_scld[u][v] != 0 and cnr_img[u + 1][v - 1] != 0:
                    cnr_img[u + 1][v - 1] = 0
                    blank[u + 1][v - 1] = 255
                    match = match + 1
        M.append(match)
        fct.append(round(factor, 3))
        print("Matches: ", match)

    N = key_points_n(img)
    repeatability = []
    for i in range(len(M)):
        repeatability.append(M[i]/N)
    print("Repeatability: ", repeatability)
    plt.figure(2)
    plt.plot(fct, repeatability, color="blue", marker=".")
    plt.xticks(fct, rotation=90)
    plt.yticks(repeatability)
    plt.xlabel("Scaling factor")
    plt.ylabel("Match frequency")
    plt.title("Scaled graph")
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.waitforbuttonpress()


def key_points_n(img):
    img_cnr = corner(img)
    img1 = img_cnr.copy()
    img2 = img_cnr.copy()

    count_n = []
    (r_height, r_width) = img_cnr.shape[:2]
    blank = np.zeros((r_height, r_width), np.uint8)
    match = 0
    for u in range(1, 477):
        for v in range(1, 637):
            if img1[u][v] != 0 and img2[u][v] != 0:
                img2[u][v] = 0
                blank[u][v] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u - 1][v] != 0:
                img2[u - 1][v] = 0
                blank[u - 1][v] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u - 1][v - 1] != 0:
                img2[u - 1][v - 1] = 0
                blank[u - 1][v - 1] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u][v - 1] != 0:
                img2[u][v - 1] = 0
                blank[u][v - 1] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u + 1][v] != 0:
                img2[u + 1][v] = 0
                blank[u + 1][v] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u + 1][v + 1] != 0:
                img2[u + 1][v + 1] = 0
                blank[u + 1][v + 1] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u - 1][v + 1] != 0:
                img2[u - 1][v + 1] = 0
                blank[u - 1][v + 1] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u][v + 1] != 0:
                img2[u][v + 1] = 0
                blank[u][v + 1] = 255
                match = match + 1
            elif img1[u][v] != 0 and img2[u + 1][v - 1] != 0:
                img2[u + 1][v - 1] = 0
                blank[u + 1][v - 1] = 255
                match = match + 1
    count_n.append(match)
    print("Key points in original image: ", match)
    return match


if __name__ == "__main__":main()