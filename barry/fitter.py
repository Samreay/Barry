import logging


class Fitter(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = None
        self.data = None

    def set_models(self, *models):
        self.models = models

    def set_data(self, *data):
        self.data = data

