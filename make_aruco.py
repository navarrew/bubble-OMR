# make_aruco_markers.py
import cv2 as cv

aruco = cv.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for id_ in [0,1,2,3,4,5]:
    img = aruco.generateImageMarker(DICT, id_, 200)
    cv.imwrite(f"aruco_{id_}.png", img)
    