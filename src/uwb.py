from helpers.data_exporter import export_and_eval
from helpers.data_reader import get_data_gs
from helpers.data_types import Datasets
from helpers.sentence_pair import SentencePair
from utils.dictionaries import ignore_list, uk_to_us, autocorrect
from utils.math import cosine_similarity
from operator import add
import re
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
from typing import Optional
from nltk.stem import WordNetLemmatizer
import nltk

# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
lemmastory = set()


def delete_axis(array, index: int, axis: int):
    if axis == 0:
        array.pop(index)
    if axis == 1:
        for inner in array:
            inner.pop(index)

    return array


def get_word_vector(vectors: dict, word: str) -> list[float] | None:
    def _try_with(_word: str):
        if _word is None:
            return None
        # _base = lemmatizer.lemmatize(_word)
        _base = _word
        if _base != _word:
            lemmastory.add((_word, _base))
        return vectors[_base] if _base in vectors else None

    candidates = [word, uk_to_us.get(word), autocorrect.get(word)]
    return next((_try_with(cand) for cand in candidates if _try_with(cand) is not None), None)


def get_word_vectors(vectors: dict, word: str, unrecognized_words: set[str] = None) -> list[list[float]]:
    vector = get_word_vector(vectors, word)
    if vector is not None:
        return [vector]

    # Handle case like 'non-bathingsuit'
    # This actually reduced score for STSint.testoutput.images.wa from 0.8766 to 0.8760
    corrected_word = autocorrect[word] if word in autocorrect else word
    # corrected_word = word
    word_parts = [word_part for word_part in re.split('[^A-Za-z]', corrected_word) if word_part]
    # print(f'{word} -> {word_parts}')

    vecs = []
    for word_part in word_parts:
        vector = get_word_vector(vectors, word_part)
        if vector is not None:
            vecs.append(vector)
        elif word_part not in ignore_list and unrecognized_words is not None:
            unrecognized_words.add(word_part)
    return vecs


def main():
    train_data: list[SentencePair] = [*get_data_gs('train', Datasets.H), *get_data_gs('train', Datasets.I),
                                      *get_data_gs('train', Datasets.AS)]
    test_data: list[SentencePair] = [*get_data_gs('test', Datasets.H), *get_data_gs('test', Datasets.I),
                                     *get_data_gs('test', Datasets.AS)]
    all_data = [*test_data, *train_data]

    data = test_data

    # data = data[:4]

    # data: list[SentencePair] = [*get_data_gs('test', Datasets.H)]
    # data: list[SentencePair] = [*get_data_gs('train', Datasets.H)]

    # for pair in data:
    #     ali = min(len(pair.sent_1.chunks), len(pair.sent_2.chunks))
    #     no_ali = max(len(pair.sent_1.chunks), len(pair.sent_2.chunks))
    #     first_longer = len(pair.sent_1.chunks) > len(pair.sent_2.chunks)
    #     pair.alignments = [*[([i], [i]) for i in range(ali)],
    #                        *[(([i], []) if first_longer else ([], [i])) for i in range(ali, no_ali)]]
    #
    # export_and_eval(data)

    # print(data[0].sent_1)

    # print(list(gensim.downloader.info()['models'].keys()))
    vectors = gensim.downloader.load('word2vec-google-news-300')
    # print(vectors['car'])

    # TODO: after align, try to add one more chunk
    # THR
    # Value for vec comp
    THR = 0.35
    # Value for lex sem vecs
    # THR = 0.45

    unrecognized_words = set()

    for pair in data:
        print(f'\n>>> PAIR #{pair.id[2] + 1}')

        # vvv vector composition
        for sent in [pair.sent_1, pair.sent_2]:
            for chunk in sent.chunk_data:
                vec = []
                for word in chunk['words']:
                    vec.extend(get_word_vectors(
                        vectors, word, unrecognized_words))
                # doesn't matter
                chunk['vec'] = np.mean(vec, axis=0) if vec else None
                # chunk['vec'] = np.sum(vec, axis=0) if vec else None

        sims = [[(-1, -1, -1) for _ in range(len(pair.sent_2.chunks))]
                for _ in range(len(pair.sent_1.chunks))]

        for i, chunk_1 in enumerate(pair.sent_1.chunk_data):
            for j, chunk_2 in enumerate(pair.sent_2.chunk_data):
                sim = cosine_similarity(chunk_1["vec"], chunk_2["vec"])
                sims[i][j] = (i, j, sim)
                print(
                    f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')
        # ^^^ vector composition

        # vvv lexical semantic vectors
        # sims = [[(-1, -1, -1) for _ in range(len(pair.sent_2.chunks))]
        #         for _ in range(len(pair.sent_1.chunks))]
        #
        # for i, chunk_1 in enumerate(pair.sent_1.chunk_data):
        #     for j, chunk_2 in enumerate(pair.sent_2.chunk_data):
        #         words = {*chunk_1["words"], *chunk_2["words"]}
        #         word_vecs = [vec for word in words for vec in get_word_vectors(vectors, word)]
        #         chunk_1_word_vecs = [vec for word in chunk_1["words"] for vec in get_word_vectors(vectors, word)]
        #         chunk_2_word_vecs = [vec for word in chunk_2["words"] for vec in get_word_vectors(vectors, word)]
        #
        #         chunk_1_vec = []
        #         chunk_2_vec = []
        #         for word_vec in word_vecs:
        #             chunk_1_vec.append(max([cosine_similarity(word_vec, chunk_1_word_vec) for chunk_1_word_vec in
        #                                     chunk_1_word_vecs] or [0]))
        #             chunk_2_vec.append(max([cosine_similarity(word_vec, chunk_2_word_vec) for chunk_2_word_vec in
        #                                     chunk_2_word_vecs] or [0]))
        #
        #         sim = cosine_similarity(chunk_1_vec, chunk_2_vec)
        #         sims[i][j] = (i, j, sim)
        #         print(
        #             f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')
        # ^^^ lexical semantic vectors

        sent_1_chids = set([i for i in range(len(pair.sent_1.chunks))])
        sent_2_chids = set([i for i in range(len(pair.sent_2.chunks))])

        # print(sims)

        # print(sims[0][1])
        # print(type(sims[0][1]))
        # print(sims[0][1][2])
        # print(len(sims[0][1]))

        while len(sims) and len(sims[0]):
            max_val = (-1, -1, -1)
            max_ij = (-1, -1)
            for i in range(len(sims)):
                for j in range(len(sims[0])):
                    if sims[i][j][2] > max_val[2]:
                        max_val = sims[i][j]
                        max_ij = (i, j)
            if max_val[2] < THR:
                break

            print(
                f'{max_val} => {" ".join(pair.sent_1.chunk_data[max_val[0]]["words"])} <=> {" ".join(pair.sent_2.chunk_data[max_val[1]]["words"])}')
            sent_1_chids.remove(max_val[0])
            sent_2_chids.remove(max_val[1])
            pair.alignments.append(([max_val[0]], [max_val[1]]))
            sims = delete_axis(sims, max_ij[0], 0)
            sims = delete_axis(sims, max_ij[1], 1)

        # fun fact vvv doesn't matter
        for chid in sent_1_chids:
            pair.alignments.append(([chid], []))
        for chid in sent_2_chids:
            pair.alignments.append(([], [chid]))

        print(pair.alignments)

    export_and_eval(data)

    print(unrecognized_words)
    print(f'Unrecognized words: {len(unrecognized_words)}')

    # print(lemmastory)


if __name__ == '__main__':
    main()
