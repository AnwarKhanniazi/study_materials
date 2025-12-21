import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import os


def convolution2d(image, kernel):
    return signal.convolve2d(image, kernel, boundary='symm', mode='same')
    # m, n = kernel.shape
    # if (m == n):
    #     y, x = image.shape
    #     y = y - m + 1
    #     x = x - m + 1
    #     new_image = np.zeros((y, x))
    #     for i in range(y):
    #         for j in range(x):
    #             new_image[i][j] = np.sum(image[i:i + m, j:j + m] * kernel)
    # return new_image


def rounded_teta(teta):
    x, y = teta.shape
    for i in range(x):
        for j in range(y):
            if teta[i][j] <= 22.5 and teta[i][j] >= -22.5:
                teta[i][j] = 0
            elif teta[i][j] <= 67.5 and teta[i][j] >= 22.5:
                teta[i][j] = 45
            elif teta[i][j] <= 112.5 and teta[i][j] >= 67.5:
                teta[i][j] = 90
            elif teta[i][j] <= 157.5 and teta[i][j] >= 112.5:
                teta[i][j] = 135
            elif teta[i][j] > 157.5:
                teta[i][j] = 180
            elif teta[i][j] >= -67.5 and teta[i][j] <= -22.5:
                teta[i][j] = -45
            elif teta[i][j] >= -112.5 and teta[i][j] <= -67.5:
                teta[i][j] = -90
            elif teta[i][j] >= -157.5 and teta[i][j] <= -112.5:
                teta[i][j] = -135
            elif teta[i][j] < -157.5:
                teta[i][j] = -180
    return teta


def find_boundary(g, teta):
    shp = g.shape
    boundary = np.zeros((shp[0], shp[1]))
    for i in range(1, shp[0] - 1):
        for j in range(1, shp[1] - 1):
            if (teta[i][j] >= 0 and teta[i][j] <= 45) or (teta[i][j] >= -180 and teta[i][j] <= -135):
                teta_rad = teta[i][j] * np.pi / 180
                r = np.tan(teta_rad) * g[i + 1][j + 1] + (1 - np.tan(teta_rad)) * g[i][j + 1]
                p = np.tan(teta_rad) * g[i - 1][j - 1] + (1 - np.tan(teta_rad)) * g[i][j - 1]
                if g[i][j] >= p and g[i][j] >= r:
                    boundary[i][j] = g[i][j]
            elif (teta[i][j] > 45 and teta[i][j] <= 90) or (teta[i][j] >= -135 and teta[i][j] <= -90):
                teta_rad = teta[i][j] * np.pi / 180
                cot = np.cos(teta_rad) / np.sin(teta_rad)
                r = cot * g[i + 1][j + 1] + (1 - cot) * g[i + 1][j]
                p = cot * g[i - 1][j - 1] + (1 - cot) * g[i - 1][j]
                if g[i][j] >= p and g[i][j] >= r:
                    boundary[i][j] = g[i][j]
            elif (teta[i][j] > 90 and teta[i][j] <= 135) or (teta[i][j] >= -90 and teta[i][j] <= -45):
                teta_rad = (teta[i][j] - 90) * np.pi / 180
                tan = np.tan(teta_rad)
                r = tan * g[i + 1][j - 1] + (1 - tan) * g[i + 1][j]
                p = tan * g[i - 1][j + 1] + (1 - tan) * g[i - 1][j]
                if g[i][j] >= p and g[i][j] >= r:
                    boundary[i][j] = g[i][j]
            elif (teta[i][j] > 135 and teta[i][j] <= 180) or (teta[i][j] >= -45 and teta[i][j] <= 0):
                teta_rad = (180 - teta[i][j]) * np.pi / 180
                tan = np.tan(teta_rad)
                r = tan * g[i + 1][j - 1] + (1 - tan) * g[i][j - 1]
                p = tan * g[i - 1][j + 1] + (1 - tan) * g[i][j + 1]
                if g[i][j] >= p and g[i][j] >= r:
                    boundary[i][j] = g[i][j]

    return boundary


def double_threshold(g, th, tl):
    x, y = g.shape
    g_th = np.zeros((x, y))
    g_tl = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if g[i][j] > tl:
                g_tl[i][j] = g[i][j]
            if g[i][j] > th:
                g_th[i][j] = g[i][j]
    return g_th, g_tl


def hythesis(img_th, img_tl):
    x, y = img_th.shape
    final_appx = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if img_tl[i][j] != 0:
                if ((img_th[i + 1, j - 1] != 0) or (img_th[i + 1, j] != 0) or (img_th[i + 1, j + 1] != 0)
                        or (img_th[i, j - 1] != 0) or (img_th[i, j + 1] != 0) or (img_th[i - 1, j - 1] != 0)
                        or (img_th[i - 1, j] != 0) or (img_th[i - 1, j + 1] != 0)):
                    final_appx[i][j] = img_tl[i][j]
            if img_th[i][j] != 0:
                final_appx[i][j] = img_th[i][j]
    return final_appx


def plot_data(image, smooth_img, abs_g, boundary, img_thsh_tl, img_thsh_th, final_edges):
    plt.imshow(image, cmap='gray')
    plt.title("Orginal Image")
    plt.axis("off")
    plt.show()
    plt.imshow(smooth_img, cmap='gray')
    plt.title("Smoothed Image")
    plt.axis("off")
    plt.show()
    plt.imshow(abs_g, cmap='gray')
    plt.title("Gradient Magnitude")
    plt.axis("off")
    plt.show()
    plt.imshow(boundary, cmap='gray')
    plt.title("Non Maximum Supression")
    plt.axis("off")
    plt.show()
    plt.imshow(img_thsh_th, cmap='gray')
    plt.title("thresholding high")
    plt.axis("off")
    plt.show()
    plt.imshow(img_thsh_tl, cmap='gray')
    plt.title("thresholding low")
    plt.axis("off")
    plt.show()
    plt.imshow(final_edges, cmap='gray')
    plt.title("Final Edges")
    plt.axis("off")
    plt.show()

    path = "D:/STUDY/UOL2025/CV_Course/Task/Exp6/img/result/"
    cv2.imwrite(os.path.join(path, 'orginal_image.jpg'), image)
    cv2.imwrite(os.path.join(path, 'smoothed_image.jpg'), smooth_img)
    cv2.imwrite(os.path.join(path, 'gradient_magnitude.jpg'), abs_g)
    cv2.imwrite(os.path.join(path, 'Non_Maximum_Supression.jpg'), boundary)
    cv2.imwrite(os.path.join(path, 'thresholding_low.jpg'), img_thsh_tl)
    cv2.imwrite(os.path.join(path, 'thresholding_high.jpg'), img_thsh_th)
    cv2.imwrite(os.path.join(path, 'final_edges.jpg'), final_edges)

def main():
    # Read Image
    image = cv2.imread("D:/STUDY/UOL2025/CV_Course/Task/Exp6/img/orginal_image.jpg", cv2.IMREAD_GRAYSCALE)

    # Smoothing by Gaussian Filter
    smooth_kernel = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5],
                              [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]], np.float32) / 159
    smooth_img = convolution2d(image, smooth_kernel)

    # Gradient Sobel Filter
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    G_x = convolution2d(smooth_img, gx_kernel)
    G_y = convolution2d(smooth_img, gy_kernel)

    # abs Of Gradient in x direction and y
    abs_g = np.sqrt(np.add(np.power(G_x, 2), np.power(G_y, 2)))
    teta = np.arctan2(G_y, G_x) * 180 / np.pi

    # Non Maximum Supression Algorithm
    boundary = find_boundary(abs_g, teta)

    # Thresholding
    T_H = np.max(boundary) * 0.25
    T_L = T_H * 0.25
    img_thsh_th, img_thsh_tl = double_threshold(boundary, T_H, T_L)

    # hythesis Algorithm
    final_edges = hythesis(img_thsh_th, img_thsh_tl)

    # plot results
    plot_data(image, smooth_img, abs_g, boundary, img_thsh_tl, img_thsh_th, final_edges)


if "__main__":
    main()
