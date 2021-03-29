"""! @brief This module contains utilities used by the other models. """
##
# @file Utils.py
#
# @brief This module contains utitilites used by the other modules, such as
#         cutting ellipses from images or getting the sharpness of an image
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
import math
import imutils
import cv2
import csv
import yaml
import argparse

_woman_names = None

__cfg = None
def get_config(attribute):
    global __cfg
    """! Returns an object containing the project configuration
    """

    if __cfg is None:
        with open('config.yaml') as f:
            __cfg = yaml.load(f, Loader=yaml.FullLoader)

    if attribute in __cfg:
        return __cfg[attribute]
    else:
        raise("Unknown config attribute: '%s'" % (attribute,))
    return None


def get_args():
    """! Prepares and parses the command arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--noPreview", help="Run without producing any visual output", action="store_true")
    parser.add_argument("--video", help="Do not run on camera, use provided video file instead")
    return parser.parse_args()

def loadWomanNames():
    """! Loads the bee names from 'Namen/Namen.list' and returns them as list
    """
    global _woman_names
    if type(_woman_names) == type(None):
        _woman_names = []
        with open('Namen/Namen.list') as _file:
            _woman_names = _file.readlines()

        # Remove newline character
        _woman_names = [line[:-1] for line in _woman_names]
    return _woman_names

def variance_of_laplacian(image):
    """! Compute the Laplacian of the image and returns a numeric value
    representing the sharpness of the image
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def cutEllipseFromImage(el, img, pad, scale=1):
    """! Cuts an ellipse from an given image and rotates it to 0 degree.
    The it calculates the sharpness of the resulting image and finaly
    it returns both, the image and the sharpness value
    @param  el      The cv2 ellipse to cut from the image
    @param  img     The image to cut the ellipse from
    @param  pad     The padding to use when cutting the ellipse
    @param  scale   The scale factor when interpreting the given ellipse
    @return  tuple  (image,sharpness)
    """

    # Scale the ellipse coordinates
    x = int(el[0]*scale)
    y = int(el[1]*scale)
    angle = el[4]

    # Get desired width/height
    w = h = 0
    if get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_150x300":
        w = 150
        h = 300
    elif get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_75x150":
        w = 75
        h = 150
    else:
        raise("Unknown setting for EXT_RES_75x150, expected EXT_RES_150x300 or EXT_RES_75x150")

    # Calcuate the size of an image the covers the rotated ellipse
    ga = (math.pi) / 180 * angle
    xb = int(math.sqrt(
            math.pow(w,2)*math.pow(math.cos(ga),2)+
            math.pow(h,2)*math.pow(math.sin(ga),2)
            ))
    yb = int(math.sqrt(
            math.pow(w,2)*math.pow(math.sin(ga),2)+
            math.pow(h,2)*math.pow(math.cos(ga),2)
            ))

    # Calculate the resulting coordinates if the above
    # rectangle gets applied to the actual image
    pc_1 =  (int(x-(xb/2)), int(y-(yb/2)))
    pc_2 =  (int(x+(xb/2)), int(y+(yb/2)))
    pc_1a = (int(x-xb), int(y-yb))
    pc_2a = (int(x+xb), int(y+yb))

    # Return None, if we are out of image borders
    if pc_1[0] < 0 or pc_1[0] > img.shape[1]:
        return None, None
    if pc_2[0] < 0 or pc_2[0] > img.shape[1]:
        return None, None
    if pc_1[1] < 0 or pc_1[1] > img.shape[0]:
        return None, None
    if pc_2[1] < 0 or pc_2[1] > img.shape[0]:
        return None, None

    # Try to crop the original image to the calculated rectangle size
    # that covers the rotated ellipse and then rotate it back to 0 degrees
    try:
        crop_img1 = img[pc_1a[1]:pc_2a[1], pc_1a[0]:pc_2a[0]].copy()
        crop_img2 = imutils.rotate_bound(crop_img1, -angle)

        crop_value = 0.4
        s0 = int((crop_img2.shape[0] -h + crop_value * h) / 2)
        s1 = int((crop_img2.shape[1] -w + crop_value * w) / 2)
    except:
        return None, None

    # Get the center of the resulting iamge to perform sharpness tests
    crop_cnt = crop_img2[s0:crop_img2.shape[0]-s0, s1:crop_img2.shape[1]-s1]
    crop_cnt = cv2.resize(crop_cnt, (45, 90))

    # Calculate a numeric value representing the image sharpness
    v = variance_of_laplacian(crop_cnt)

    # Crop the image to the desired size
    s0 = int((crop_img2.shape[0] -h)/2)
    s1 = int((crop_img2.shape[1] -w)/2)
    crop_img3 = crop_img2[s0:crop_img2.shape[0]-s0, s1:crop_img2.shape[1]-s1]
    crop_img3 = crop_img3[0:h, 0:w]

    return crop_img3, v


# All credits to Ajasja from stackoverflow!
# https://stackoverflow.com/questions/7946187/point-and-ellipse-rotated-position-test-algorithm
def pointInEllipse(p, e):
    """! Returns to if the given point (p) is inside of the given ellipse (e)
    """

    # Coordinates of the point
    xp = p[0]
    yp = p[1]

    # Center coordinates of the ellipse
    xe = e[0][0]
    ye = e[0][1]

    # Diameters of the ellipse
    rex = e[1][0] / 2
    rey = e[1][1] / 2

    # Angle converted to degrees
    angle = e[2] / 180 * math.pi # ((e[2] * 180 / math.pi) + 180) % 180

    # Pre calculate cos/sin
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Returns a value <= 1 if the point is inside of the ellipse
    t1 = cos_a*(xp - xe) + sin_a*(yp - ye)
    t2 = sin_a*(xp - xe) - cos_a*(yp - ye)
    res = ((t1*t1)/(rex*rex)) + ((t2*t2)/(rey * rey))
    return  res <= 1


def get_frame_config():
    """! Returns a configuration for the image provider on how
         to prepare and provide the captured frames
    """
    frame_config = None
    if get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_75x150":
        frame_config = (
                        (540, 960, cv2.IMREAD_UNCHANGED),
                        (180, 320,  cv2.IMREAD_UNCHANGED)
                    )
    elif get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_150x300":
        frame_config = (
                        (1080, 1920, cv2.IMREAD_UNCHANGED),
                        (540, 960, cv2.IMREAD_UNCHANGED),
                        (180, 320,  cv2.IMREAD_UNCHANGED)
                    )
    else:
        raise BaseException("Wrong image extraction setting")

    return frame_config
