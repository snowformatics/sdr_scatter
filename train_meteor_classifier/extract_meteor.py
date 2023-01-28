import cv2
import skimage
from skimage.filters import try_all_threshold, threshold_triangle
import os
from skimage import img_as_ubyte, img_as_uint
import numpy as np
from operator import itemgetter

# Extract meteor ROIs from Spectrum Lab Screenshots
# For Test purposes use the two captures in images folder

path_source = "E:/sdr/meteors/04/captures/"
path_roi = "E:/sdr/meteors/rois4/"
l = os.listdir(path_source)
# Loop over all images
for i in l:
    if i.endswith('.jpg'):
        org_image = cv2.imread(path_source + i)
        img = skimage.io.imread(path_source + i)
        R = img[..., 0]

        # Try all possibles thresholds
        #fig, ax = try_all_threshold(R, figsize=(10, 8), verbose=False)
        # Triangle currently works best
        thresh = threshold_triangle(R)
        # Binary image
        binary = R > thresh
        binary = img_as_uint(binary)

        # Extract contours
        contours_filtered = []
        rects = []
        contours, hierarchy = cv2.findContours(img_as_ubyte(binary), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(org_image, contours, -1, (0, 255, 0), 3)
        #cv2.imshow("", org_image)
        #cv2.waitKey(0)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:
                mask = np.zeros(img.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                ave_color = cv2.mean(img, mask=mask)[:3]
                m = sum(list(ave_color)) / len(list(ave_color))

                # We use a simple RGB thresholding to filter the contours
                if ave_color[0] > 15 and ave_color[0] < 85 and ave_color[1] > 4 and ave_color[1] < 35 and ave_color[2] > 60 and ave_color[2] < 90:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    padding = 30
                    contours_filtered.append((y- padding, y + h + padding, x - padding, x + w + padding, area))

                elif ave_color[0] > 80 and ave_color[0] < 135 and ave_color[1] > 30 and ave_color[1] < 65 and ave_color[2] > 40 and ave_color[2] < 75:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    padding = 30
                    contours_filtered.append((y- padding, y + h + padding, x - padding, x + w + padding, area))

        contours_filtered = sorted(contours_filtered, reverse=True, key=itemgetter(4))

        padding = 10
        for c in contours_filtered[0:1]:
            ROI = org_image[c[0] :c[1], c[2]:c[3]]
            try:
                cv2.imwrite(path_roi + i, ROI)

            except:
                print ('Can not extract ROI')
