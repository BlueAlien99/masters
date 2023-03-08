from helpers.data_exporter import export_and_eval
from helpers.data_reader import get_data_gs
from helpers.data_types import Datasets
from helpers.sentence_pair import SentencePair
from operator import add
import re
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
from typing import Optional


def cos_sim(a: list[float] | None, b: list[float] | None):
    if a is None or b is None:
        return 0

    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return min(max(0, sim), 1)


def delete_axis(array, index: int, axis: int):
    if axis == 0:
        array.pop(index)
    if axis == 1:
        for inner in array:
            inner.pop(index)

    return array


def get_word_vectors(vectors: dict, word: str, unrecognized_words: Optional[set[str]]) -> list[list[float]]:
    if word in vectors:
        return [vectors[word]]

    word_parts = [word_part for word_part in re.split(
        '[^A-Za-z]', word) if word_part]
    print(f'{word} -> {word_parts}')

    vecs = []
    for word_part in word_parts:
        if word_part in vectors:
            vecs.append(vectors[word_part])
        else:
            unrecognized_words.add(word_part)
    return vecs



def main():
    # data: list[SentencePair] = [*get_data_gs('test', Datasets.H), *get_data_gs('test', Datasets.I),
    #                             *get_data_gs('test', Datasets.AS),
    #                             *get_data_gs('train', Datasets.H), *get_data_gs('train', Datasets.I),
    #                             *get_data_gs('train', Datasets.AS)]
    #
    # data = data[:4]

    data: list[SentencePair] = [*get_data_gs('test', Datasets.H)]

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

    # TODO: unrecognized_words UK to US
    # TODO: after align, try to add one more chunk
    # TODO: THR
    THR = 0.3

    unrecognized_words = set()

    for pair in data:
        print(f'\n>>> PAIR #{pair.id[2]}')
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
                sim = cos_sim(chunk_1["vec"], chunk_2["vec"])
                sims[i][j] = (i, j, sim)
                print(
                    f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')

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



if __name__ == '__main__':
    main()
