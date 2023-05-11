import os
from typing import Literal

import gensim.downloader
from gensim.models import KeyedVectors

import helpers.paths as paths


Model = Literal['fasttext-wiki-news-subwords-300',
                'conceptnet-numberbatch-17-06-300',
                'word2vec-ruscorpora-300', 'word2vec-google-news-300',
                'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300',
                'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200']


def get_available_models():
    return list(gensim.downloader.info()['models'].keys())


def load_keyed_vectors(model: Model = 'word2vec-google-news-300') -> dict:
    kv_path = f'{paths.gensim_data_path}/{model}_kv'

    if os.path.isfile(kv_path):
        return KeyedVectors.load(kv_path, mmap='r')

    kv = gensim.downloader.load(model)
    kv.save(kv_path)

    return kv  # type: ignore
