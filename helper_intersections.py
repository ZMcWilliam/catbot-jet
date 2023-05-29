import numpy as np
import cv2
# None for nothing happening
# 2-* For two white contours
# 3-* For three white contours
# 4-* For four white contours
# *-ng For no green detected
# *-rg For green detected and turning right
# *-lg For green detected and turning left
#Black contour value
#cv2.contourArea(contour), cv2.minAreaRect(contour), contour, 0
#Green contour value
#cv2.contourArea(contour), cv2.minAreaRect(contour), contour
#White contour values 
#cv2.contourArea(contour), cv2.minAreaRect(contour, cv2.approxPolyDP(contour)), True), contour
def GetYFromX(m, c, x):
    if (m is None):
        return None
    return m*x+c
def GetXFromY(m, c, y):
    if (m is None):
        return None
    return (y-c)/m
def GetLineEquation(p1, p2):
    if (p1[0] > p2[0]):
        p1,p2=p2,p1
    if (p1[0] == p2[0]):
        return None, p1[0]
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    c = p1[1]-m*p1[0]
    return m,c
def Distance(p1, p2):
    return np.linalg.norm(p1-p2)
YFromX = np.vectorize(GetYFromX)

def CutMaskWithLine(p1, p2, mask, direction):
    # try:
    if (p1[0] > p2[0]):
        p1,p2=p2,p1
    m,c = GetLineEquation(p1, p2)
    if (m is None):
        if (direction == "left"):
            mask[:, :p1[0]] = 255
        else:
            mask[:, p2[0]:] = 255
        return mask
    # print(m,c)
    p1 = [int(GetXFromY(m, c, 0)), 0]
    p2 = [int(GetXFromY(m, c, mask.shape[0])), mask.shape[0]]
    print("p1", p1, p2)
    if (direction == "right"):
        contour = np.array([p1, [mask.shape[1], 0], [mask.shape[1], mask.shape[0]], p2]).astype(int)
    else:
        contour = np.array([p1, [0, 0], [0, mask.shape[0]], p2]).astype(int)
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(new_mask, [contour], 0, 255, -1)
    new_mask = cv2.bitwise_not(new_mask)
    new_mask = cv2.bitwise_or(mask, new_mask)
    return new_mask

# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
# TYSM GRUNDRIG!!!!!
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
# Return true if line segments AB and CD intersect
def intersect(line1,line2):
    A = line1[0]
    B = line1[1]
    C = line2[0]
    D = line2[1]
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)