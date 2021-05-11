import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from singleton import Singleton
import logging

@Singleton
class Logger(object):
    def __init__(self):
        #logging.basicConfig(filename='', encoding='utf-8', level=logging.DEBUG)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        return

    def info(self, message):
        logging.info(message)
        return

    def error(self, message):
        logging.error(message)
        return
