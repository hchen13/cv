import os

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.join(BASE_DIR, 'library')
QUERY_DIR = os.path.join(BASE_DIR, 'query-image')

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

KMEANS_MAX_ITER = 400