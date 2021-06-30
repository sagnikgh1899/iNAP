# import the necessary packages
import cv2 as cv
import numpy as np
import os
import math
from validclust import dunn
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time


def Capturing_and_Extracting_Fingernails():
    def stackImages(scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                   None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                            scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

    def MarkPoints_Right(image):
        finger_contous = [

            # Leftmost contour of hand
            (100, 470), (100, 460), (100, 450), (100, 440), (100, 430), (100, 420), (100, 410), (100, 400) \
            , (100, 390), (100, 380), (100, 370), (100, 360), (100, 350), (100, 340), (100, 330), (100, 320) \
            , (100, 310), (100, 300), (100, 290), (100, 280), (100, 270), (100, 260), (100, 250), (100, 240) \
            , (100, 230), (100, 220) \
 \
            # Topmost contour of hand
            , (100, 210), (110, 200), (120, 190), (130, 180), (140, 170), (150, 160), (160, 150), (170, 140), (180, 130) \
            , (190, 120), (200, 110), (210, 110), (220, 110), (230, 110), (240, 110), (250, 110), (260, 110) \
            , (270, 110), (280, 110), (290, 110), (300, 110), (310, 110), (320, 110), (330, 110), (340, 110) \
            , (350, 110), (360, 110), (370, 110), (380, 110), (390, 110), (400, 110), (410, 110), (420, 110) \
            , (430, 110), (440, 110), (450, 110), (460, 110), (470, 110), (480, 110), (490, 110), (500, 110) \
 \
            # Rightmost contour of hand
            , (505, 120), (510, 130), (515, 140), (520, 150), (525, 160), (530, 170), (530, 180), (530, 190) \
            , (530, 200), (530, 210), (530, 220), (530, 230), (530, 240), (530, 250), (530, 260), (530, 270) \
            , (530, 280), (530, 290), (530, 300), (530, 310), (530, 320), (530, 330), (530, 340), (530, 350) \
            , (530, 360), (520, 370), (510, 380), (500, 390), (490, 400), (480, 410), (470, 420), (460, 430) \
            , (450, 440), (440, 450), (430, 460) \
 \
            # Bottommost contour of hand
            , (420, 470), (410, 470), (400, 470), (390, 470), (380, 470), (370, 470), (360, 470), (350, 470) \
            , (340, 470), (330, 470), (320, 470), (310, 470), (300, 470), (290, 470), (280, 470), (270, 470) \
            , (260, 470), (250, 470), (240, 470), (230, 470), (220, 470), (210, 470), (200, 470), (190, 470) \
            , (180, 470), (170, 470), (160, 470), (150, 470), (140, 470), (130, 470), (120, 470), (110, 470) \
 \
            # Small finger left
            , (180, 370), (180, 360), (180, 350), (180, 340), (180, 330), (180, 320), (180, 310), (180, 300),
            (180, 290),
            (180, 280) \
            , (180, 270), (180, 260), (180, 250), (180, 240), (180, 230), (180, 220), (180, 210), (180, 200) \
            , (180, 190), (180, 180), (180, 170), (180, 160), (180, 150), (180, 140), (180, 130) \
 \
            # Small finger right
            , (240, 120), (240, 130), (240, 140), (240, 150), (240, 160), (240, 170), (240, 180), (240, 190) \
            , (240, 200), (240, 210), (240, 220), (240, 230), (240, 240), (240, 250), (240, 260), (240, 270) \
            , (240, 280), (240, 290), (240, 300), (240, 310), (240, 320), (240, 330), (240, 340), (240, 350) \
            , (240, 360), (240, 370) \
 \
            # Ring finger right
            , (300, 360), (300, 350), (300, 340), (300, 330), (300, 320), (300, 310), (300, 300), (300, 290) \
            , (300, 280), (300, 270), (300, 260), (300, 250), (300, 240), (300, 230), (300, 220), (300, 210) \
            , (300, 200), (300, 190), (300, 180), (300, 170), (300, 160), (300, 150), (300, 140), (300, 130), (300, 120) \
 \
            # Middle finger right
            , (360, 360), (360, 350), (360, 340), (360, 330), (360, 320), (360, 310), (360, 300), (360, 290) \
            , (360, 280), (360, 270), (360, 260), (360, 250), (360, 240), (360, 230), (360, 220), (360, 210) \
            , (360, 200), (360, 190), (360, 180), (360, 170), (360, 160), (360, 150), (360, 140), (360, 130), (360, 120) \
 \
            # Index finger right
            , (420, 370), (420, 360), (420, 350), (420, 340), (420, 330), (420, 320), (420, 310), (420, 300), (420, 290) \
            , (420, 280), (420, 270), (420, 260), (420, 250), (420, 240), (420, 230), (420, 220), (420, 210) \
            , (420, 200), (420, 190), (420, 180), (420, 170), (420, 160), (420, 150), (420, 140), (420, 130), (420, 120) \
 \
            # Upper limit of nail
            , (190, 310), (200, 310), (210, 310), (220, 310), (230, 310), (240, 310), (250, 310), (260, 310), (270, 310) \
            , (280, 310), (290, 310), (300, 310), (310, 310), (320, 310), (330, 310), (340, 310), (350, 310) \
            , (360, 310), (370, 310), (380, 310), (390, 310), (400, 310), (410, 310) \
 \
            # Lower limit of nail
            , (190, 370), (200, 370), (210, 370), (220, 370), (230, 370), (240, 370), (250, 370), (260, 370) \
            , (270, 370), (280, 370), (290, 370), (300, 370), (310, 370), (320, 370), (330, 370), (340, 370) \
            , (350, 370), (360, 370), (370, 370), (380, 370), (390, 370), (400, 370), (410, 370)

        ]

        radius = 1
        color = (0, 255, 255)
        thickness = 2

        for i in range(len(finger_contous)):
            cv.circle(image, finger_contous[i], radius, color, thickness)

        """for i in range(len(finger_contous) - 1):
            cv.line(image, finger_contous[i], finger_contous[i + 1], (0, 0, 0), 1)
        for i in range(len(top_contours) - 1):
            cv.line(image, top_contours[i], top_contours[i + 1], (0, 0, 0), 1)"""

        # cv.imshow("res", image)
        # cv.waitKey(0)

    def MarkPoints_Left(image):
        finger_contous = [

            # Leftmost contour of hand
            (555, 470), (555, 460), (555, 450), (555, 440), (555, 430), (555, 420), (555, 410), (555, 400) \
            , (555, 390), (555, 380), (555, 370), (555, 360), (555, 350), (555, 340), (555, 330), (555, 320) \
            , (555, 310), (555, 300), (555, 290), (555, 280), (555, 270), (555, 260), (555, 250), (555, 240) \
            , (555, 230), (555, 220) \
 \
            # Topmost contour of hand
            , (150, 160), (155, 150), (160, 140), (165, 130), (170, 120), (180, 110), (190, 110) \
            , (200, 110), (210, 110), (220, 110), (230, 110), (240, 110), (250, 110), (260, 110) \
            , (270, 110), (280, 110), (290, 110), (300, 110), (310, 110), (320, 110), (330, 110), (340, 110) \
            , (350, 110), (360, 110), (370, 110), (380, 110), (390, 110), (400, 110), (410, 110), (420, 110) \
            , (430, 110), (440, 110), (455, 120), (465, 130), (475, 140), (485, 150), (495, 160) \
            , (505, 170), (515, 180), (525, 190), (535, 200), (545, 210) \
 \
            # Rightmost contour of right hand
            , (145, 170), (145, 180), (145, 190) \
            , (145, 200), (145, 210), (145, 220), (145, 230), (145, 240), (145, 250), (145, 260), (145, 270) \
            , (145, 280), (145, 290), (145, 300), (145, 310), (145, 320), (145, 330), (145, 340), (145, 350) \
            , (145, 360), (145, 370), (155, 380), (165, 390), (175, 400), (185, 410), (195, 420), (205, 430) \
            , (215, 440), (225, 450), (235, 460) \
 \
            # Bottommost contour of hand
            , (555, 470), (545, 470), (535, 470), (525, 470), (515, 470), (505, 470), (495, 470), (485, 470) \
            , (475, 470), (465, 470), (455, 470), (445, 470), (435, 470), (425, 470), (415, 470), (405, 470) \
            , (395, 470), (385, 470), (375, 470), (365, 470), (355, 470), (345, 470), (335, 470), (325, 470) \
            , (315, 470), (305, 470), (295, 470), (285, 470), (275, 470), (265, 470), (255, 470), (245, 470) \
 \
            # Index finger left
            , (210, 370), (210, 360), (210, 350), (210, 340), (210, 330), (210, 320), (210, 310), (210, 300) \
            , (210, 290), (210, 280), (210, 270), (210, 260), (210, 250), (210, 240), (210, 230), (210, 220) \
            , (210, 210), (210, 200), (210, 190), (210, 180), (210, 170), (210, 160), (210, 150), (210, 140) \
            , (210, 130), (210, 120) \
 \
            # Index finger right
            , (270, 120), (270, 130), (270, 140), (270, 150), (270, 160), (270, 170), (270, 180), (270, 190) \
            , (270, 200), (270, 210), (270, 220), (270, 230), (270, 240), (270, 250), (270, 260), (270, 270) \
            , (270, 280), (270, 290), (270, 300), (270, 310), (270, 320), (270, 330), (270, 340), (270, 350) \
            , (270, 360), (270, 370) \
 \
            # Middle finger right
            , (330, 360), (330, 350), (330, 340), (330, 330), (330, 320), (330, 310), (330, 300), (330, 290) \
            , (330, 280), (330, 270), (330, 260), (330, 250), (330, 240), (330, 230), (330, 220), (330, 210) \
            , (330, 200), (330, 190), (330, 180), (330, 170), (330, 160), (330, 150), (330, 140), (330, 130), (330, 120) \
 \
            # Ring finger right
            , (390, 360), (390, 350), (390, 340), (390, 330), (390, 320), (390, 310), (390, 300), (390, 290) \
            , (390, 280), (390, 270), (390, 260), (390, 250), (390, 240), (390, 230), (390, 220), (390, 210) \
            , (390, 200), (390, 190), (390, 180), (390, 170), (390, 160), (390, 150), (390, 140), (390, 130), (390, 120) \
 \
            # Small finger right
            , (450, 370), (450, 360), (450, 350), (450, 340), (450, 330), (450, 320), (450, 310), (450, 300), (450, 290) \
            , (450, 280), (450, 270), (450, 260), (450, 250), (450, 240), (450, 230), (450, 220), (450, 210) \
            , (450, 200), (450, 190), (450, 180), (450, 170), (450, 160), (450, 150), (450, 140), (450, 130), (450, 120) \
 \
            # Upper limit of nail
            , (220, 310), (230, 310), (240, 310), (250, 310), (260, 310), (270, 310) \
            , (280, 310), (290, 310), (300, 310), (310, 310), (320, 310), (330, 310), (340, 310), (350, 310) \
            , (360, 310), (370, 310), (380, 310), (390, 310), (400, 310), (410, 310), (420, 310), (430, 310) \
            , (440, 310) \
 \
            # Lower limit of nail
            , (220, 370), (230, 370), (240, 370), (250, 370), (260, 370) \
            , (270, 370), (280, 370), (290, 370), (300, 370), (310, 370), (320, 370), (330, 370), (340, 370) \
            , (350, 370), (360, 370), (370, 370), (380, 370), (390, 370), (400, 370), (410, 370), (420, 370) \
            , (430, 370), (440, 370) \
 \
            ]

        radius = 1
        color = (0, 255, 255)
        thickness = 2
        for i in range(len(finger_contous)):
            cv.circle(image, finger_contous[i], radius, color, thickness)

        """for i in range(len(finger_contous) - 1):
            cv.line(image, finger_contous[i], finger_contous[i + 1], (0, 0, 0), 1)
        for i in range(len(top_contours) - 1):
            cv.line(image, top_contours[i], top_contours[i + 1], (0, 0, 0), 1)"""

        # cv.imshow("res", image)
        # cv.waitKey(0)

    def Crop_Image(img, hand):
        if hand == 'L':
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            indexfinger_left = np.array([[(220, 312), (230, 312), (240, 312), (250, 312), (260, 312), (270, 312) \
                                             , (268, 320), (268, 330), (268, 340), (268, 350), (268, 360), (268, 370) \
                                             , (260, 368), (250, 368), (240, 368), (230, 368), (220, 368) \
                                             , (222, 360), (222, 350), (222, 340), (222, 330), (222, 320)

                                          ]])

            middlefinger_left = np.array(
                [[(272, 370), (272, 360), (272, 350), (272, 340), (272, 330), (272, 320), (272, 310) \
                     , (280, 312), (290, 312), (300, 312), (310, 312), (320, 312), (330, 312) \
                     , (328, 320), (328, 330), (328, 340), (328, 350), (328, 360), (328, 370) \
                     , (320, 368), (310, 368), (300, 368), (290, 368), (280, 368)
                  ]])

            ringfinger_left = np.array(
                [[(332, 370), (332, 360), (332, 350), (332, 340), (332, 330), (332, 320), (332, 310) \
                     , (340, 312), (350, 312), (360, 312), (370, 312), (380, 312), (390, 312) \
                     , (388, 320), (388, 330), (388, 340), (388, 350), (388, 360), (388, 370) \
                     , (380, 368), (370, 368), (360, 368), (350, 368), (340, 368)
                  ]])

            smallfinger_left = np.array(
                [[(392, 370), (392, 360), (392, 350), (392, 340), (392, 330), (392, 320), (392, 310) \
                     , (400, 312), (410, 312), (420, 312), (430, 312), (440, 312), (450, 312) \
                     , (448, 320), (448, 330), (448, 340), (448, 350), (448, 360), (448, 370) \
                     , (440, 368), (430, 368), (420, 368), (410, 368), (400, 368)
                  ]])

            cv.drawContours(mask, [indexfinger_left], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(indexfinger_left)  # returns (x,y,w,h) of the rect
            cropped_indexfinger_left = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_indexfinger_left)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.drawContours(mask, [middlefinger_left], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(middlefinger_left)  # returns (x,y,w,h) of the rect
            cropped_middlefinger_left = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_middlefinger_left)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.drawContours(mask, [ringfinger_left], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(ringfinger_left)  # returns (x,y,w,h) of the rect
            cropped_ringfinger_left = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_ringfinger_left)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.drawContours(mask, [smallfinger_left], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(smallfinger_left)  # returns (x,y,w,h) of the rect
            cropped_smallfinger_left = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_smallfinger_left)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.imwrite("smallfinger.png", cropped_smallfinger_left)
            cv.imwrite("ringfinger.png", cropped_ringfinger_left)
            cv.imwrite("middlefinger.png", cropped_middlefinger_left)
            cv.imwrite("indexfinger.png", cropped_indexfinger_left)

            imgStack = stackImages(1, (
                [cropped_smallfinger_left, cropped_ringfinger_left, cropped_middlefinger_left,
                 cropped_indexfinger_left]))

            cv.imshow("Stacked Images", imgStack)  ## Uncomment this
            cv.waitKey(0)
            cv.destroyAllWindows()


        elif hand == 'R':
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)

            smallfinger_right = np.array(
                [[(182, 370), (182, 360), (182, 350), (182, 340), (182, 330), (182, 320), (182, 310) \
                     , (190, 312), (200, 312), (210, 312), (220, 312), (230, 312) \
                     , (238, 310), (238, 320), (238, 330), (238, 340), (238, 350), (238, 360), (238, 370) \
                     , (230, 368), (220, 368), (210, 368), (200, 368), (190, 368)
                  ]])

            ringfinger_right = np.array([[
                (242, 310), (242, 320), (242, 330), (242, 340), (242, 350), (242, 360), (242, 370) \
                , (250, 368), (260, 368), (270, 368), (280, 368), (290, 368), (300, 368) \
                , (298, 360), (298, 350), (298, 340), (298, 330), (298, 320), (298, 310) \
                , (250, 312), (260, 312), (270, 312), (280, 312), (290, 312)

            ]])

            middlefinger_right = np.array([[
                (302, 310), (302, 320), (302, 330), (302, 340), (302, 350), (302, 360), (302, 370) \
                , (310, 368), (320, 368), (330, 368), (340, 368), (350, 368), (360, 368) \
                , (358, 360), (358, 350), (358, 340), (358, 330), (358, 320), (358, 310) \
                , (310, 312), (320, 312), (330, 312), (340, 312), (350, 312), (360, 312)

            ]])

            indexfinger_right = np.array([[
                (362, 310), (362, 320), (362, 330), (362, 340), (362, 350), (362, 360) \
                , (370, 368), (380, 368), (390, 368), (400, 368), (410, 368), (420, 368) \
                , (418, 360), (418, 350), (418, 340), (418, 330), (418, 320), (418, 310) \
                , (370, 312), (380, 312), (390, 312), (400, 312), (410, 312)

            ]])

            # method 1 smooth region
            cv.drawContours(mask, [smallfinger_right], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(smallfinger_right)  # returns (x,y,w,h) of the rect
            cropped_smallfinger_right = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            # crate the white background of the same size of original image
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            # overlap the resulted cropped image on the white background
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_smallfinger_right)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.drawContours(mask, [ringfinger_right], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(ringfinger_right)  # returns (x,y,w,h) of the rect
            cropped_ringfinger_right = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_ringfinger_right)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.drawContours(mask, [middlefinger_right], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(middlefinger_right)  # returns (x,y,w,h) of the rect
            cropped_middlefinger_right = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_middlefinger_right)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.drawContours(mask, [indexfinger_right], -1, (255, 255, 255), -1, cv.LINE_AA)
            res = cv.bitwise_and(img, img, mask=mask)
            rect = cv.boundingRect(indexfinger_right)  # returns (x,y,w,h) of the rect
            cropped_indexfinger_right = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            wbg = np.ones_like(img, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            dst = wbg + res
            # cv.imshow('Original', img)
            # cv.imshow("Mask", mask)
            # cv.imshow("Cropped", cropped_indexfinger_right)
            # cv.imshow("Samed Size Black Image", res)
            # cv.imshow("Samed Size White Image", dst)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            cv.imwrite("smallfinger.png", cropped_smallfinger_right)
            cv.imwrite("ringfinger.png", cropped_ringfinger_right)
            cv.imwrite("middlefinger.png", cropped_middlefinger_right)
            cv.imwrite("indexfinger.png", cropped_indexfinger_right)

            imgStack = stackImages(1, (
                [cropped_smallfinger_right, cropped_ringfinger_right, cropped_middlefinger_right,
                 cropped_indexfinger_right]))

            cv.imshow("Stacked Images", imgStack)  ## Uncomment this
            cv.waitKey(0)
            cv.destroyAllWindows()

    # Fingers of which hand to be captured
    which_hand = str(input("L for Left Hand \t R for Right Hand\n"))

    # Video camera Capture

    cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        if which_hand.upper() == 'L':
            MarkPoints_Left(frame)
        elif which_hand.upper() == 'R':
            MarkPoints_Right(frame)
        else:
            print("Incorrect Entry!!")
            break
        cv.imshow('frame', frame)
        img_counter = 0

        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "Frame_{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    img = cv.imread('Frame_0.png')
    Crop_Image(img, which_hand.upper())


# Type_of_Input = str(input("Manual(M) / Automatic(A)\t"))
# if Type_of_Input.upper() == 'A':
#     Capturing_and_Extracting_Fingernails()
#     start_time = time.time()
# else:
#     start_time = time.time()


######################################################################################################


#######################################################################################################
def isbright(image, thresh=0.3):
    # Resize image
    image = cv.resize(image, (400, 400))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)
    # Return True if mean is greater than thresh else False
    print("T =", np.mean(L))
    return np.mean(L) > thresh


# High Dynamic Range
def auto_adjust_gamma(img, img_counter, inp):
    # METHOD 1: RGB

    # convert img to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # compute gamma
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid * 255) / math.log(mean)
    print("Gamma =", gamma)

    # do gamma correction
    img_gamma1 = np.power(img, gamma).clip(0, 255).astype(np.uint8)
    img_name = "autogamma_image_{}.png".format(img_counter)
    cv.imwrite(img_name, img_gamma1)

    if inp.upper() == 'N':
        img_gamma1 = img
        img_name = "autogamma_image_{}.png".format(img_counter)
        cv.imwrite(img_name, img_gamma1)

    # cv.imwrite('autogamma_image.png', img_gamma1)


# Program starts from here
Type_of_Input = str(input("Manual(M) / Automatic(A)\t"))

if Type_of_Input.upper() == 'A':
    Capturing_and_Extracting_Fingernails()
    start_time = time.time()

    img1 = cv.imread('smallfinger.png')
    auto_gamma_yes_or_no = "N" if isbright(img1) else "Y"
    print("Auto_Gamma =", auto_gamma_yes_or_no)
    auto_adjust_gamma(img1, 1, auto_gamma_yes_or_no)
    img2 = cv.imread('ringfinger.png')
    auto_gamma_yes_or_no = "N" if isbright(img2) else "Y"
    print("Auto_Gamma =", auto_gamma_yes_or_no)
    auto_adjust_gamma(img2, 2, auto_gamma_yes_or_no)
    img3 = cv.imread('middlefinger.png')
    auto_gamma_yes_or_no = "N" if isbright(img3) else "Y"
    print("Auto_Gamma =", auto_gamma_yes_or_no)
    auto_adjust_gamma(img3, 3, auto_gamma_yes_or_no)
    img4 = cv.imread('indexfinger.png')
    auto_gamma_yes_or_no = "N" if isbright(img4) else "Y"
    print("Auto_Gamma =", auto_gamma_yes_or_no)
    auto_adjust_gamma(img4, 4, auto_gamma_yes_or_no)

else:
    finger_count = int(input("Enter the number of fingers\t"))
    print("\n")
    start_time = time.time()

    i = 1
    while i <= finger_count:
        img_name = 'Image_{}.png'.format(i)
        img = cv.imread(img_name)
        auto_gamma_yes_or_no = "N" if isbright(img) else "Y"
        print("Auto_Gamma =", auto_gamma_yes_or_no)
        auto_adjust_gamma(img, i, auto_gamma_yes_or_no)
        i += 1

#######################################################################################################


#######################################################################################################
# Circle Region Extraction
def Circle_Region_Extraction(img_path, img_counter):
    def Circular_Extraction(image):
        row, column = len(image), len(image[0])
        # print(row//2, column//2)
        center = (column // 2, row // 2)
        radius = 20
        color = [0, 0, 0]
        thickness = -1
        cv.circle(image, center, radius, color, thickness)
        Masked_image_path = "Masked.png"
        cv.imwrite(Masked_image_path, image)
        return Masked_image_path

    img = cv.imread(img_path)
    masked_img_path = Circular_Extraction(img)

    new_img = cv.imread(img_path)
    masked_img = cv.imread(masked_img_path)
    sub_img = cv.subtract(new_img, masked_img)

    img_name = "Nailbed_image_{}.png".format(img_counter)
    cv.imwrite(img_name, sub_img)
    # cv.imwrite("Nailbed_image_{}.png".format(img_counter))"Nailbed.png", sub_img)
    # cv.imshow("img", sub_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    os.remove(masked_img_path)


img1_path = 'autogamma_image_1.png'
Circle_Region_Extraction(img1_path, 1)
img2_path = 'autogamma_image_2.png'
Circle_Region_Extraction(img2_path, 2)
img3_path = 'autogamma_image_3.png'
Circle_Region_Extraction(img3_path, 3)
if Type_of_Input.upper() == 'A' or finger_count > 3:
    img4_path = 'autogamma_image_4.png'
    Circle_Region_Extraction(img4_path, 4)


#######################################################################################################


#######################################################################################################
# Adaptive K Generation and Kmeans Dominant Color
def Adaptive_K_Value(image):
    data = image / 255.0  # use 0...1 scale
    data = data.reshape(image.shape[0] * image.shape[1], 3)
    # print(data.shape)

    # X = np.array(data)
    data = np.array(data)

    max_Dunn = float("-inf")
    min_DB = float("inf")
    max_Silhoutte = float("-inf")
    max_Calinski = float("-inf")

    ideal_k = k = 2

    while True:
        print("k = ", k + 1)

        kmeans = KMeans(n_clusters=k).fit(data)
        kmeans_labels = kmeans.labels_
        # print(kmeans.labels_)
        # print(kmeans.cluster_centers_)

        distances = pairwise_distances(data)
        Dunn = dunn(distances, kmeans_labels)
        print("Dunn Index kmeans =", Dunn)
        DB = davies_bouldin_score(data, kmeans_labels)
        print("DB Index kmeans =", DB)
        Silhoutte = silhouette_score(data, kmeans_labels)
        print('Silhouette Score kmeans=', Silhoutte)
        Calinski = metrics.calinski_harabasz_score(data, kmeans_labels)
        print('Calinski-Harabasz Index kmeans =', Calinski)

        count = 0
        if max_Dunn < Dunn:
            count += 1
        if min_DB > DB:
            count += 1
        if max_Silhoutte < Silhoutte:
            count += 1
        if max_Calinski < Calinski:
            count += 1
        if count < 2:
            break
        else:
            ideal_k = k
            k += 1
            max_Dunn = Dunn
            min_DB = DB
            max_Silhoutte = Silhoutte
            max_Calinski = Calinski

    print("\nOptimum K value =", ideal_k + 1)
    return ideal_k


def Kmeans_Dominant_Color(image_path, K):
    image_BRG = cv.imread(image_path)
    image_RGB = cv.cvtColor(image_BRG, cv.COLOR_BGR2RGB)

    # cv.imshow("Image", image_RGB)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    image_reshaped = image_RGB.reshape((image_RGB.shape[0] * image_RGB.shape[1], 3))
    # print(image_reshaped)

    clt = KMeans(n_clusters=K)  # + 1)
    clt.fit(image_reshaped)

    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        # return the histogram
        return hist

    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                         color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    def Dominant_Color(hist, centroids):
        Max_percent = float("-inf")
        max_val = max(hist)
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            if Max_percent < percent != max_val:
                Max_percent = percent
                dom_color = color.astype("uint8").tolist()
        return dom_color

    hist = centroid_histogram(clt)
    # print(hist)
    # print(clt.cluster_centers_)
    bar = plot_colors(hist, clt.cluster_centers_)
    Dominant_color = Dominant_Color(hist, clt.cluster_centers_)
    print("Dominant Color = ", Dominant_color)
    # return Dominant_color
    # print(bar[0])
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    return Dominant_color


Store_RGB_Values = []

print("\n**********************************************************")
image1 = cv.imread('Nailbed_image_1.png')
ideal_k_1 = Adaptive_K_Value(image1)
Store_RGB_Values.append(Kmeans_Dominant_Color("Nailbed_image_1.png", ideal_k_1 + 1))

print("\n**********************************************************")
image2 = cv.imread('Nailbed_image_2.png')
ideal_k_2 = Adaptive_K_Value(image2)
Store_RGB_Values.append(Kmeans_Dominant_Color("Nailbed_image_2.png", ideal_k_1 + 1))

print("\n**********************************************************")
image3 = cv.imread('Nailbed_image_3.png')
ideal_k_3 = Adaptive_K_Value(image3)
Store_RGB_Values.append(Kmeans_Dominant_Color("Nailbed_image_3.png", ideal_k_3 + 1))

if Type_of_Input.upper() == 'A' or finger_count > 3:
    print("\n**********************************************************")
    image4 = cv.imread('Nailbed_image_4.png')
    ideal_k_4 = Adaptive_K_Value(image4)
    Store_RGB_Values.append(Kmeans_Dominant_Color("Nailbed_image_4.png", ideal_k_4 + 1))
#######################################################################################################


#######################################################################################################
# Hemoglobin Measurement
print("\n\nRGBs =", Store_RGB_Values)
b_val = []
g_val = []
r_val = []

for val in Store_RGB_Values:  # Since image is in BGR format
    # print(val)
    b_val.append(val[2])
    g_val.append(val[1])
    r_val.append(val[0])

# print(r_val, g_val, b_val)
avg_value_r = sum(r_val) / len(r_val)
avg_value_g = sum(g_val) / len(g_val)
avg_value_b = sum(b_val) / len(b_val)
print(avg_value_r, avg_value_g, avg_value_b)

val1 = (-1.922 + 0.206 * avg_value_r - 0.241 * avg_value_g + 0.012 * avg_value_b)
Num = math.exp(val1)
Den = 1 + math.exp(val1)
add_val = 0.05
L = Num / Den + add_val
# print("Actual L =", L)
if round(L * 10, 2) < 5:
    L = 8.5 + round(L * 10, 2)
    if L < 6:
        print("\n Hgb =",math.ceil(L+1),"g/dl")
    else:
        print("\n Hgb =",L,"g/dl")
else:
    if round(L * 10, 2) + L < 6:
        print("\n Hgb = ", round(L * 10, 2) + L + 1.2, "g/dl")
    else:
        print("\n Hgb = ", round(L * 10, 2) + L, "g/dl")  # L is the addition factor because of the regression formula

if L < 11.5:  # A Threshold value I have used on the basis of data available to me
    print("\nAnemic")
else:
    print("\nNon-Anemic")

print("\n\n--- %s seconds ---" % (time.time() - start_time))

##############################################################################################
# Removal of images
os.remove('autogamma_image_1.png')
os.remove('autogamma_image_2.png')
os.remove('autogamma_image_3.png')
os.remove('Nailbed_image_1.png')
os.remove('Nailbed_image_2.png')
os.remove('Nailbed_image_3.png')

if Type_of_Input.upper() == 'M':
    if finger_count > 3:
        os.remove('autogamma_image_4.png')
        os.remove('Image_1.png')
        os.remove('Image_2.png')
        os.remove('Image_3.png')
        os.remove('Image_4.png')
        os.remove('Nailbed_image_4.png')
elif Type_of_Input.upper() == 'A':
    os.remove('smallfinger.png')
    os.remove('ringfinger.png')
    os.remove('indexfinger.png')
    os.remove('middlefinger.png')
    os.remove('Nailbed_image_4.png')

#############################################################################################
