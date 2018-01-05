import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import json

from matching.core import *
from matching.utils import *


def one_pass():
    size = (256, 256)
    tick = datetime.now()
    print("Loading image...")
    library_images = load_library(resize=size, gray=True)
    tock = datetime.now()
    print("Loading takes {}\n".format(tock - tick))

    k = guess_k(library_images, algorithm='sift')
    print("Features will be grouped into {} clusters\n".format(k))

    tick = datetime.now()
    print("Preprocessing...")
    codebook, idf, lookup = preprocess(library_images, k=k)
    tock = datetime.now()
    print("Preprocessing takes {}\n".format(tock - tick))

    query_files = get_image_files(QUERY_DIR)
    corrects = get_correct_matches()

    exact_match, approx_match = 0, 0
    top = 3
    for name, path in query_files:
        sample_image = load_image(path, gray=True, resize=size)
        tick = datetime.now()
        sparse_vector = calculate_sparse_vector(sample_image, codebook, idf)
        similarities = calculate_cosine_similarity(sparse_vector, lookup)
        result_vec = (-similarities).argsort()
        results = result_vec[0, :top].tolist()
        tock = datetime.now()
        print("Matching takes {}\n".format(tock - tick))

        correct_matches = corrects[name]
        for i, r in enumerate(results):
            if r in correct_matches:
                approx_match += 1
                if i == 0:
                    exact_match += 1

        # matched_images = [library_images[i] for i in results]
        # display_image(sample_image, *matched_images, col=4)
    total = len(query_files)
    print("Exact hit accuracy: {:.2f}%\tTop-{} hit accuracy: {:.2f}%\n".
          format(exact_match / total * 100, top, approx_match / total * 100))


if __name__ == '__main__':
    one_pass()

    # library = load_library(gray=False)
    # corrects = get_correct_matches()
    # for query_file, query_path in get_image_files(QUERY_DIR):
    #     image = load_image(query_path, gray=False)
    #     correct_images = [library[i] for i in corrects[query_file]]
    #     display_image(image, *correct_images, col=1 + len(correct_images))
