import os
import random
import re

import gensim.downloader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

import helpers.paths as paths


def get_available_corpora():
    # ['semeval-2016-2017-task3-subtaskBC', 'semeval-2016-2017-task3-subtaskA-unannotated', 'patent-2017',
    #  'quora-duplicate-questions', 'wiki-english-20171001', 'text8', 'fake-news', '20-newsgroups']

    return list(gensim.downloader.info()['corpora'].keys())


def load_tfidf_model(frac: float):
    corpora = 'wiki-english-20171001'
    frac_str = str(round(frac, 3))

    dict_path = f'{paths.gensim_data_path}/{corpora}_{frac_str}_dict'
    tfidf_path = f'{paths.gensim_data_path}/{corpora}_{frac_str}_tfidf'

    if os.path.isfile(dict_path) and os.path.isfile(tfidf_path):
        dct = Dictionary.load(dict_path)
        tfidf = TfidfModel.load(tfidf_path)

        return dct, tfidf

    dataset = gensim.downloader.load(corpora)
    dataset_len = gensim.downloader.info(corpora)["num_records"]

    data = []
    progress = 0

    print('Transforming words...')
    for i, article in enumerate(dataset):
        new_progress = round(i*100 / dataset_len, 1)
        if new_progress != progress:
            progress = new_progress
            print(f'{progress}%', end='\r')
        if random.random() > frac:
            continue
        words = [word for word in re.split('[^A-Za-z-]+', ' '.join(article['section_texts'])) if word]
        data.append(words)

    print('Fitting dictionary...')
    dct = Dictionary(data)
    dct.save(dict_path)

    print('Converting corpus to BoW format...')
    corpus = [dct.doc2bow(line) for line in data]
    print('Fitting Tfidf model...')
    tfidf = TfidfModel(corpus)
    tfidf.save(tfidf_path)

    return dct, tfidf
