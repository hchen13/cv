from unittest import TestCase

import numpy as np

from matching.settings import *


class TestUtils(TestCase):

    def setUp(self):
        self.correct_path = os.path.join(LIBRARY_DIR, 'capitol1.jpg')
        self.non_exist_path = os.path.join(LIBRARY_DIR, 'capital1.jpg')
        self.non_image_path = os.path.join(BASE_DIR, 'tests.py')
        self.correct_dir = LIBRARY_DIR
        self.non_exist_dir = os.path.join(BASE_DIR, 'somenonexistingdir')

    def test_is_image(self):
        from matching.utils import is_image
        assert True == is_image(self.correct_path)
        assert False == is_image(self.non_exist_path)
        assert False == is_image(self.non_image_path)

    def test_load_image(self):
        from matching.utils import load_image
        result = load_image(self.correct_path)
        self.assertIsInstance(result, np.ndarray)
        result = load_image(self.non_exist_path)
        self.assertIsNone(result)
        result = load_image(self.non_image_path)
        self.assertIsNone(result)

    def test_load_images(self):
        from matching.utils import load_images
        result = load_images(self.correct_dir)
        self.assertIsInstance(result, list)
        for i in result:
            self.assertIsInstance(i, np.ndarray)
        result = load_images(self.non_exist_dir)
        self.assertIsInstance(result, list)
        self.assertIs(len(result), 0)

    def test_get_image_files(self):
        from matching.utils import get_image_files
        result = get_image_files(self.correct_dir)
        self.assertIsInstance(result, list)
        for i in result:
            self.assertIsInstance(i, tuple)
            self.assertEqual(len(i), 2)
        result = get_image_files(self.non_exist_dir)
        self.assertIsNone(result)
