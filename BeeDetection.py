from Config import *
import cv2
import logging
import numpy as np
import math

logger = logging.getLogger(__name__)


def detect_bees(frame, scale):

    # Helper method to calculate distance between to ellipses
    def near(p1,p2):
        return math.sqrt(math.pow(p1[0]-p2[0], 2) + math.pow(p1[1]-p2[1], 2))

    # Helper method to calculate the area of an ellipse
    def area(e1):
        return np.pi * e1[1][0] * e1[1][1]

    # Extract BGR and HSV channels
    b,g,r = cv2.split(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    # Substract G and V
    o = 255 - (g - v)

    # Blur Image and perform a binary thresholding
    o = cv2.GaussianBlur(o, (9,9), 9)
    _, o = cv2.threshold(o, BINARY_THRESHOLD_VALUE, BINARY_THRESHOLD_MAX, cv2.THRESH_BINARY)

    # Invert result
    o = 255 -o

    # Detect contours
    contours, hierarchy = cv2.findContours(o, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    ellipses = []
    groups = []
    for i in range(len(contours)):

        # Only countours with more than five edges can fit an ellipse
        if(len(contours[i]) >= 5):

            # Fit ellipse
            e = cv2.fitEllipse(contours[i])

            # Skip too small detections
            if e[1][0] < 8 or e[1][1] < 8:
                continue

            # Only use ellipses with minium size
            ellipseArea = area(e)
            if ellipseArea > DETECT_ELLIPSE_AREA_MIN_SIZE and ellipseArea < DETECT_ELLIPSE_AREA_MAX_SIZE:

                # Scale ellipse to desired size
                e = ((e[0][0] * scale, e[0][1] * scale), (e[1][0] * scale, e[1][1] * scale), e[2])
                ellipses.append(e)
            elif ellipseArea > DETECT_GROUP_AREA_MIN_SIZE  and ellipseArea < DETECT_GROUP_AREA_MAX_SIZE:

                # Scale ellipse to desired size
                e = ((e[0][0] * scale, e[0][1] * scale), (e[1][0] * scale, e[1][1] * scale), e[2])
                groups.append(e)

    # Merge nearby detection into one
    done = []
    skip = []
    solved = []
    for a in ellipses:

        # Find ellipses that are close to each other and store them as a group
        group = []
        for b in ellipses:

            # Skip self and already processed ellipes
            if (a,b) in done or (b,a) in done or a == b:
                continue
            done.append((a,b))

            # Calculate distance between both ellipses
            dist = near(a[0],b[0])
            if dist < 50:

                # Put them into the group
                if a not in group:
                    group.append(a)
                if b not in group:
                    group.append(b)

                # Remember which ellipses were processed
                if not a in skip:
                    skip.append(a)
                if not b in skip:
                    skip.append(b)

        # Isolate the ellipse with the biggest area
        if len(group):
            solved.append(max(group, key=area))

    # Merge isolated ellipses with remaining ones
    rest = list(filter(lambda x: x not in skip, ellipses))
    merged = rest + solved

    return merged, groups

