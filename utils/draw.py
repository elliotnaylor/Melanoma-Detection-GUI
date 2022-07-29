import math as m
import numpy as np
import cv2

#http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
def bres_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    steep = abs(dy) > abs(dx)

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    #Recalculate
    dx = x2 - x1
    dy = y2 - y1

    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    if swapped:
        points.reverse()
    return points

def drawX(image, center, rotation):

    #0 degrees to 90 degrees 
    theta = np.arange(0, m.pi, m.pi/2)

    radius = 100
    theta_180 = m.pi

    #Calculate trajectory of line from center
    for t in theta:

        #Start of skin lesion area
        point1 = (int(center[0] + radius * m.cos(t+rotation)), 
                  int(center[1] - radius * m.sin(t+rotation)))

        #Opposite side of skin lesion area
        point2 = (int(center[0] + radius * m.cos(t+theta_180+rotation)),
                 int(center[1] - radius * m.sin(t+theta_180+rotation)))

        #lines.append(bres_line(point1, point2))
        cv2.line(image, point1, point2, (255, 255, 255), 1)

#https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
#Should be rewritten in C++ cuda for faster processing
def rotate_bound(image, angle):
    #Determine centerpoint of image
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    #Find nerw boundry for image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    #Adjust rotation to account for translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    #Perform rotation
    return cv2.warpAffine(image, M, (nW, nH))