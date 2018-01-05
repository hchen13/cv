from sklearn.cluster import KMeans

from .utils import *


def root_sift(des, eps=1e-7):
    """
    rootSIFT optimization to SIFT descriptors.
    References:
    1. https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
    2. https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/presentation.pdf

    :param des: SIFT descriptors represented as a numpy ndarray
    :param eps: a small value epsilon needed by rootSIFT algorithm
    :return: rootSIFT result as a numpy ndarray
    """
    if des is None:
        return None
    des /= (des.sum(axis=1, keepdims=True) + eps)
    return np.sqrt(des)


def extract_features(images, algorithm='sift'):
    """
    Extract SIFT or SURF keypoints and descriptors

    :param images: a list of numpy ndarrays each of which represents an image
    :param algorithm: a string 'sift' or 'surf' representing the algorithm
    :return:
    """
    kp_set, des_set = [], []
    if not isinstance(images, list):
        images = [images]
    detector = sift if algorithm == 'sift' else surf
    for img in images:
        kp, des = detector.detectAndCompute(img, None)
        if not kp:
            print("Image does not have any keypoints extracted from {} algorithm,"
                  "try a higher resolution".format(algorithm))
            continue
        des = root_sift(des)
        kp_set.append(kp)
        des_set.append(des)
    return kp_set, des_set


def get_features_count(images, algorithm='sift'):
    _, des_set = extract_features(images, algorithm)
    total = 0
    for i in des_set:
        total += i.shape[0]
    return total


def guess_k(images, algorithm='sift'):
    count = get_features_count(images, algorithm)
    m = len(images)
    average_n = count / m
    theta = np.log(average_n) / 2
    number_of_features_per_image = int(average_n // theta)
    return m * number_of_features_per_image


def kmeans(mat, k=2, max_iter=KMEANS_MAX_ITER):
    """
    Clustering row vectors in a matrix into 'k' classes using k-means algorithm

    :param mat: numpy ndarray representing the matrix to be clustered
    :param k: the number of clusters
    :param max_iter: maximum number of iterations
    :return: the clustered centroids
    """
    codebook = KMeans(n_clusters=k, max_iter=max_iter)
    codebook.fit(mat)
    return codebook


def get_bow(I, codebook):
    bow = codebook.predict(I)
    size = bow.shape[0]
    return bow.reshape(1, size)


def get_distance_mat(I, A):
    """
    Given 2 matrices 'I' and 'A', calculate the distance between each row of I[i] and A[j] and form a distance matrix

    :param I: first matrix
    :param A: second matrix
    :return: distances organized as a matrix
    """
    n, k = I.shape[0], A.shape[0]

    # Euclidean distance
    I2 = np.dot(I, I.T).diagonal().reshape(n, 1)
    A2 = np.dot(A, A.T).diagonal().reshape(1, k)
    prod = -2 * np.dot(I, A.T)
    euclidean_distance = prod + A2 + I2

    # Cosine distance
    # numerator = np.dot(I, A.T)
    # denominator = np.sqrt(np.dot(I2, A2))
    # cosine_distance = 1 - numerator / denominator

    return euclidean_distance


def get_min_dist_index(D):
    """
    Return the index (0-based) of the minimum value of each row in a matrix 'D'

    :param D: matrix whose min value indices in each row to be extracted
    :return: a column vector containing the min value indices
    """
    t = D.argsort(axis=1)
    length = t.shape[0]
    q = t[:, 0].reshape(length, 1).T
    return q


def get_min_dists(D):
    """
    Calculate the min_dist vectors for all the matrices in the list

    :param D: a list of matrices
    :return: a list containing min_dist vectors for each matrix
    """
    min_dists = []
    for d in D:
        md = get_min_dist_index(d)
        min_dists.append(md)
    return min_dists


def get_sparse(bows, k):
    """
    Generate a sparse matrix,

    :param min_dists:
    :param k:
    :return:
    """
    sparse = np.zeros((len(bows), k))
    for i, md in enumerate(bows):
        for j in md[0]:
            sparse[i, j] += 1
    return sparse


def preprocess(library_images, k=None, algorithm='sift'):
    """
    Preprocess the images in the library and generate the codebook (a sklearn.clusters.KMeans object)
    inverse document frequency (IDF), and the lookup table

    :param library_images: a list of ndarrays representing the images in the library
    :param k: the number of clusters used for k-means
    :param algorithm: the algorithm used for extracting feature descriptors
    :return: codebook, idf, lookup
    """
    m = len(library_images)
    if k is None:
        k = m * 10
    _, des_set = extract_features(library_images, algorithm)
    I_all = np.vstack(des_set)
    k = min(k, I_all.shape[0])
    codebook = kmeans(I_all, k=k, max_iter=KMEANS_MAX_ITER)

    bow_list = []
    for des in des_set:
        bow = get_bow(des, codebook)
        bow_list.append(bow)

    sparse = get_sparse(bow_list, k)
    occurrence = np.sum(sparse != 0, axis=0, keepdims=True)
    idf = np.log( (m + 1) / (occurrence + 1) )
    sparse_weighted = sparse * idf
    sum_square = np.sum(np.power(sparse_weighted, 2), axis=1, keepdims=True)
    lookup = sparse_weighted / sum_square
    return codebook, idf, lookup


def calculate_sparse_vector(image, codebook, idf, algorithm='sift'):
    """
    Get the sparse vector to be looked up

    :param image: target image represented as ndarray
    :param codebook: the k-means code word vocabulary
    :param idf: inversed document frequency for each of the words in the codebook
    :param algorithm: algorithm used for extracting feature descriptors
    :return: a sparse vector representing the corresponding image's weighted bag of visual words
    """
    k = codebook.n_clusters
    _, des = extract_features(image, algorithm)
    I = des[0]
    bow = get_bow(I, codebook)
    sparse = get_sparse([bow], k)
    sparse_weighted = sparse * idf
    sum_square = np.sum(np.power(sparse_weighted, 2), axis=1, keepdims=True)
    return sparse_weighted / sum_square
