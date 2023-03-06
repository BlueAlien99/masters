from helpers.data_exporter import export_and_eval
from helpers.data_reader import get_data_gs
from helpers.data_types import Datasets
from helpers.sentence_pair import SentencePair
from operator import add
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np


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


def process_sentence(pair):
    pass


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

    # TODO: after align, try to add one more chunk
    # TODO: THR
    THR = 0.5

    for pair in data:
        print(f'>>> PAIR #{pair.id[2]}')
        for sent in [pair.sent_1, pair.sent_2]:
            for chunk in sent.chunk_data:
                # vec = None
                vec = []
                for word in chunk['words']:
                    if word in vectors:
                        # TODO: add vs avg
                        # vec = vectors[word] if vec is None else list( map(add, vec, vectors[word]) )
                        vec.append(vectors[word])
                # chunk['vec'] = vec
                chunk['vec'] = np.mean(vec, axis=0) if vec else None

        sims = [[(-1, -1, -1) for _ in range(len(pair.sent_2.chunks))] for _ in range(len(pair.sent_1.chunks))]

        for i, chunk_1 in enumerate(pair.sent_1.chunk_data):
            for j, chunk_2 in enumerate(pair.sent_2.chunk_data):
                sim = cos_sim(chunk_1["vec"], chunk_2["vec"])
                sims[i][j] = (i, j, sim)
                print(f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')

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
            print(max_val)
            print(max_val[0])
            print(max_val[1])
            print(f'{max_val} => {" ".join(pair.sent_1.chunk_data[max_val[0]]["words"])} <=> {" ".join(pair.sent_2.chunk_data[max_val[1]]["words"])}')
            sent_1_chids.remove(max_val[0])
            sent_2_chids.remove(max_val[1])
            pair.alignments.append(([max_val[0]], [max_val[1]]))
            sims = delete_axis(sims, max_ij[0], 0)
            sims = delete_axis(sims, max_ij[1], 1)

        print(pair.alignments)

        #TODO: noali

    export_and_eval(data)


if __name__ == '__main__':
    main()
