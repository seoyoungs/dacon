# 캐니로 노이즈 제거
# https://076923.github.io/posts/Python-opencv-14/

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg

# cv2.imread() 함수를 이용하여 이미지 파일을 읽습니다.
image = cv2.imread("../dacon12/data/train/00000.png", cv2.IMREAD_GRAYSCALE) # 파일 읽기 
pix = np.array(image)

# =================== 이미지 전처리==========================

#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((pix <= 254) & (pix != 0), 0, pix)

# 이미지 팽창
x_df3 = cv2.dilate(x_df2, kernel=np.ones((2, 2), np.uint8), iterations=1) 
#np.ones((2, 2) 이사이즈로 늘리기, iterations 외곽 픽셀 주변에 1(흰색)으로 추가

# 블러 적용, 노이즈 제거
x_df4 = cv2.medianBlur(src=x_df3, ksize= 5) # 블러처리 대상, 사이즈

# 이전 파일 것
# canny 가장자리 검출
canny = cv2.Canny(x_df4, 30, 70) #원본 이미지, 임계값1, 임계값2, 커널 크기, L2그라디언트
sobelx = cv2.Sobel(x_df4, cv2.CV_64F, 1, 0, ksize=3) #  가장자리 검출을 적용
sobely = cv2.Sobel(x_df4, cv2.CV_64F, 0, 1, ksize=3)
laplacian = cv2.Laplacian(x_df4, cv2.CV_8U)

images = [canny, sobelx, sobely, laplacian]
titles = ['canny', 'sobelx', 'sobely', 'laplacian']

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i]), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

