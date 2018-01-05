import re

import cv2
import numpy as np
from PIL import Image
from .settings import *
from matplotlib import pyplot as plt


def is_image(path):
    """
    determine if the given path directs to a valid image file

    :param path: file url
    :return: True if valid else False
    """
    if path.startswith('.'):
        return False
    try:
        Image.open(path)
    except Exception as e:
        # print("Invalid image file:", e)
        return False
    return True


def load_image(file_path, resize=False, gray=True):
    """
    Load an image from a file and return a numpy matrix

    :param file_path: string - image url
    :param resize: 2-tuple - the size to which the image will be resized
    :param gray: boolean - indicate whether to load the image in gray scale
    :return: ndarray - numpy matrix, None if the image file is invalid
    """
    try:
        img = Image.open(file_path)
    except Exception as e:
        print("In load_image() function:", e)
        return None
    if resize and isinstance(resize, tuple):
        img.thumbnail(resize, Image.ANTIALIAS)
    img = np.array(img)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_images(dir, resize=False, gray=True):
    """
    Load all images from a directory, skipping non-image files without warnings.
    Loading can be optionally set with a size indicating the resized image width/height

    :param dir: string - image url
    :param resize: 2-tuple - the size to which the image will be resized
    :param gray: boolean - indicate whether to load the image in gray scale
    :return: list containing all the images in `dir`, represented as ndarrays
    """
    try:
        files = os.listdir(dir)
    except FileNotFoundError as e:
        print("in load_images() function:", e)
        files = []
    image_list = []
    for file_name in files:
        path = os.path.join(dir, file_name)
        if not is_image(path):
            continue
        image_list.append(load_image(path, resize, gray))
    return image_list


def load_library(resize=False, gray=True):
    """
    Load all images in the library folder

    :param resize: same as load_images()
    :param gray: same as load_images()
    :return: list containing all the images in the library directory
    """
    return load_images(LIBRARY_DIR, resize, gray)


def display_image(*images, title='图片', col=2):
    """
    Plot the images

    :param images: arbitrary number of ndarrays each of which represents an image
    :param title: optionally set the plot title
    :param col: optionally change the number of images shown in one row, default to 2
    :return: None
    """
    plt.figure(figsize=(8, 4.5))
    plt.title(title)
    row = np.math.ceil(len(images) / col)
    for i, image in enumerate(images):
        plt.subplot(row, col, i + 1)
        plt.imshow(image)
    plt.show()


def get_image_files(dir):
    """
    Get the list of image files from `dir`, skipping all other types of files

    :param dir: directory to be extracted
    :return: list containing file names and absolute path, None if the directory not found
    """
    try:
        files = os.listdir(dir)
    except FileNotFoundError as e:
        print("in get_image_files() function:", e)
        return None
    image_files = []
    for name in files:
        path = os.path.join(dir, name)
        if not is_image(path):
            continue
        image_files.append((name, path))
    return image_files


def calculate_cosine_similarity(A, B):
    """
    The function calculates the cosine similarity between 2 matrices.
    Note that each matrix is a list of row vectors stacking up vertically where each vector is of size 'd'.

    :param A: matrix A of shape(ma, d)
    :param B: matrix B of shape(mb, d)
    :return: the cosine between each vector pairs
    """
    prod = np.dot(A, B.T)  # prod is of shape (ma, mb)
    return prod
    A2 = np.sum(A ** 2, axis=1, keepdims=True)  # A2 is of shape (ma, 1)
    B2 = np.sum(B ** 2, axis=1, keepdims=True)  # B2 is of shape (mb, 1)
    denominator = np.dot(np.sqrt(A2), np.sqrt(B2).T)  # denominator is of shape (ma, mb)
    similarities = prod / denominator
    return similarities


def get_entity_name(file_name):
    matches = re.match(r'(.*/)*(?P<name>[a-z]+)\d?\.', file_name)
    if matches:
        return matches.groupdict()['name']
    return None


def get_correct_matches():
    corrects = {}
    query_files = get_image_files(QUERY_DIR)
    library_files = get_image_files(LIBRARY_DIR)
    for query_file_name, _ in query_files:
        query_name = get_entity_name(query_file_name)
        corrects[query_file_name] = []
        for i, (library_file_name, _) in enumerate(library_files):
            library_name = get_entity_name(library_file_name)
            if query_name == library_name:
                corrects[query_file_name].append(i)
    return corrects
