import cv2
import matplotlib.pyplot as plt
import skimage
from skimage.filters import try_all_threshold, threshold_triangle
import os
from skimage import img_as_ubyte, img_as_uint
import numpy as np
from operator import itemgetter

path = "E:/sdr/meteors/01/"
l = os.listdir(path)
for i in l:

    org_image = cv2.imread(path + i)
    img = skimage.io.imread(path + i)
    R = img[..., 0]

    #fig, ax = try_all_threshold(R, figsize=(10, 8), verbose=False)
    thresh = threshold_triangle(R)
    binary = R > thresh

    #binary_image = threshold_yen(q2_channel[q2_channel != 255])
    #binary_image = q2_channel < binary_image
    binary = img_as_uint(binary)
    #kernel = np.ones((2, 2), np.uint8)
    #binary = cv2.dilate(binary, kernel, iterations=1)
    #cv2.imshow("", binary)
    #cv2.waitKey(0)
    contours_filtered = []
    rects = []
    contours, hierarchy = cv2.findContours(img_as_ubyte(binary), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 65:

            #x, y, w, h = cv2.boundingRect(cnt)  # offsets - with this you get 'mask'
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #avg_color = np.array(cv2.mean(img[y:y + h, x:x + w])).astype(np.uint8)
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            ave_color = cv2.mean(img, mask=mask)[:3]
            m = sum(list(ave_color)) / len(list(ave_color))
            #print(area, ave_color, m)
            #cv2.drawContours(org_image, [cnt], 0, (0, 255, 0), 2)
            #cv2.imshow("", org_image)
            #cv2.waitKey(0)
            if ave_color[0] > 45 and ave_color[0] < 85 and ave_color[1] > 5 and ave_color[1] < 35 and ave_color[2] > 60 and ave_color[2] < 85:
                #print(area, ave_color, m)
                (x, y, w, h) = cv2.boundingRect(cnt)
                padding = 50
                contours_filtered.append((y- padding, y + h + padding, x - padding, x + w + padding, area))
                #cv2.rectangle(org_image, (x - padding, y - padding), (x + w + padding, y + h + padding), (255, 0, 0), 2)
                #cv2.drawContours(org_image, [cnt], 0, (0, 255, 0),2)
                #cv2.imshow("", org_image)
                cv2.waitKey(0)
            elif ave_color[0] > 95 and ave_color[0] < 130 and ave_color[1] > 40 and ave_color[1] < 55 and ave_color[2] > 45 and ave_color[2] < 66:
                #print(area, ave_color, m)
                (x, y, w, h) = cv2.boundingRect(cnt)
                padding = 50
                contours_filtered.append((y- padding, y + h + padding, x - padding, x + w + padding, area))
                #cv2.rectangle(org_image, (x - padding, y - padding), (x + w + padding, y + h + padding), (255, 0, 0), 2)
                #cv2.drawContours(org_image, [cnt], 0, (0, 255, 0), 2)
                #cv2.imshow("", org_image)
                #cv2.waitKey(0)
    #print (i, contours_filtered)
    contours_filtered = sorted(contours_filtered, reverse=True, key=itemgetter(4))
    # contours_filtered2 = combine_boxes(contours_filtered)
    padding = 10
    for c in contours_filtered[0:1]:
        ROI = org_image[c[0] :c[1], c[2]:c[3]]

        print (i)
        #cv2.rectangle(org_image, (c[2] - padding, c[0] - padding), (c[3] + padding, c[1] + padding), (255, 0, 0), 2)
        cv2.imwrite("E:/sdr/meteors/02/" + i, ROI)
        #cv2.imshow("", org_image)
        #cv2.imshow("", ROI)
        #cv2.waitKey(0)
