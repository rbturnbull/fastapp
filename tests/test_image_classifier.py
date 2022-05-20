from fastapp.testing import FastAppTestCase
from fastapp.examples.image_classifier import ImageClassifier


class TestImageClassifier(FastAppTestCase):
    app_class = ImageClassifier
