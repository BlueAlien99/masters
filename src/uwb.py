from helpers.data_exporter import export_and_eval
from helpers.data_reader import get_data_gs
from helpers.data_types import Datasets
from helpers.sentence_pair import SentencePair, Alignment
from utils.dictionaries import ignore_list, uk_to_us, autocorrect
from utils.math import cosine_similarity, delete_axis
from operator import add
import re
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
from typing import Tuple
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
import gc

# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
lemmastory = set()


def get_word_vector(vectors: object, word: str) -> list[float] | None:
    def _try_with(_word: str):
        if _word is None:
            return None
        # _base = lemmatizer.lemmatize(_word)
        _base = _word
        # _base = f'/c/en/{_word}'
        # _base = _word.lower()  # OH, NO!
        if _base != _word:
            lemmastory.add((_word, _base))
        return vectors[_base] if _base in vectors else None

    # candidates = [word, uk_to_us.get(word), autocorrect.get(word)]
    # candidates = [word, uk_to_us.get(word)]
    # HOLY @#$%! <<<< +0.0055
    # candidates = [word, word.lower(), word[:-1], word.lower()[:-1], uk_to_us.get(word)]
    candidates = [word, word.lower(), word[:-1], word.lower()[:-1], uk_to_us.get(word), word[:-2], word.lower()[:-2]]
    return next((_try_with(cand) for cand in candidates if _try_with(cand) is not None), None)


def get_word_vectors(vectors: object, word: str, unrecognized_words: set[str] = None) -> list[list[float]]:
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


def get_chunk_vector(vectors: object, words: list[str], unrecognized_words: set[str] = None) -> list[float]:
    vec = [vec for word in words for vec in get_word_vectors(vectors, word, unrecognized_words)]
    # doesn't matter
    return np.mean(vec, axis=0) if vec else None
    # return np.sum(vec, axis=0) if vec else None


def get_printable_alignment(pair: SentencePair, alignment: Alignment, max_val: Tuple):
    return f'{max_val} => {" ".join(pair.sent_1.chunks_to_words(alignment[0]))} <=> {" ".join(pair.sent_2.chunks_to_words(alignment[1]))}'


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
    # ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300',
    #  'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200',
    #  'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200',
    #  '__testing_word2vec-matrix-synopsis']
    kv_data = 'word2vec-google-news-300'
    kv_local = f'gensim-data/{kv_data}'
    # vectors = gensim.downloader.load(kv_data)
    # vectors.save(kv_local)
    vectors = KeyedVectors.load(kv_local, mmap='r')
    # print(vectors['car'])

    # print(list(gensim.downloader.info()['corpora'].keys()))
    # ['semeval-2016-2017-task3-subtaskBC', 'semeval-2016-2017-task3-subtaskA-unannotated', 'patent-2017',
    #  'quora-duplicate-questions', 'wiki-english-20171001', 'text8', 'fake-news', '20-newsgroups',
    #  '__testing_matrix-synopsis', '__testing_multipart-matrix-synopsis']
    tfidf_data = 'wiki-english-20171001'
    tfidf_data = 'text8'
    tfidf_local = f'gensim-data/{tfidf_data}'
    dict_local = f'gensim-data/dict'
    dataset = gensim.downloader.load(tfidf_data)
    print('downloader')
    dct = Dictionary(dataset)  # fit dictionary
    print('dict')
    dct.save(dict_local)
    print('save dict')
    corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
    print('corpus')

    # del dataset
    # gc.collect()

    model = TfidfModel(corpus)  # fit model
    print('tfidf model')
    model.save(tfidf_local)
    # vector = model[corpus[0]]  # apply model to the first corpus document

    # print(corpus[0])
    # print(vector)

    return

    # THR
    # Value for vec comp
    THR = 0.35
    # Value for lex sem vecs
    # THR = 0.45

    unrecognized_words = set()

    for pair in data:
        print_buf = []

        print_buf.append(f'\n\n>>> PAIR #{pair.id[2] + 1}')
        print_buf.append(' <> '.join([' '.join(cd['words']) for cd in pair.sent_1.chunk_data]))
        print_buf.append(' <> '.join([' '.join(cd['words']) for cd in pair.sent_2.chunk_data]))
        print_buf.append('')

        sim_strings = []

        # vvv vector composition
        for sent in [pair.sent_1, pair.sent_2]:
            for chunk in sent.chunk_data:
                chunk['vec'] = get_chunk_vector(vectors, chunk['words'], unrecognized_words)

        sims = [[(-1, -1, -1) for _ in range(len(pair.sent_2.chunks))]
                for _ in range(len(pair.sent_1.chunks))]

        for i, chunk_1 in enumerate(pair.sent_1.chunk_data):
            for j, chunk_2 in enumerate(pair.sent_2.chunk_data):
                sim = cosine_similarity(chunk_1["vec"], chunk_2["vec"])
                sims[i][j] = (i, j, sim)
                sim_strings.append(
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
        #         sim_strings.append(
        #             f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')
        # ^^^ lexical semantic vectors

        sim_strings.sort(reverse=True)
        print_buf.append('\n'.join([*sim_strings, '']))

        sent_1_chids = set([i for i in range(len(pair.sent_1.chunks))])
        sent_2_chids = set([i for i in range(len(pair.sent_2.chunks))])

        # print_buf.append(sims)

        # print_buf.append(sims[0][1])
        # print_buf.append(type(sims[0][1]))
        # print_buf.append(sims[0][1][2])
        # print_buf.append(len(sims[0][1]))

        # while len(sims) and len(sims[0]):
        #     max_val = (-1, -1, -1)
        #     max_ij = (-1, -1)
        #     for i in range(len(sims)):
        #         for j in range(len(sims[0])):
        #             if sims[i][j][2] > max_val[2]:
        #                 max_val = sims[i][j]
        #                 max_ij = (i, j)
        #     if max_val[2] < THR:
        #         break
        #
        #     sent_1_chids.remove(max_val[0])
        #     sent_2_chids.remove(max_val[1])
        #     alignment = ([max_val[0]], [max_val[1]])
        #     pair.alignments.append(alignment)
        #     print_buf.append(get_printable_alignment(pair, alignment, max_val))
        #     sims = delete_axis(sims, max_ij[0], 0)
        #     sims = delete_axis(sims, max_ij[1], 1)
        #
        # should_print = True

        should_print = False

        while True:
            max_val = (-1, -1, -1)
            for i in range(len(sims)):
                for j in range(len(sims[0])):
                    if sims[i][j][2] > max_val[2]:
                        max_val = sims[i][j]
            if max_val[2] < THR:
                break

            chids_1_used = []
            chids_2_used = []
            for ali in pair.alignments:
                if max_val[0] in ali[0] or max_val[1] in ali[1]:
                    chids_1_used.extend([chid for chid in ali[0] if chid != max_val[0]])
                    chids_2_used.extend([chid for chid in ali[1] if chid != max_val[1]])

            sims[max_val[0]][max_val[1]] = (max_val[0], max_val[1], -1)
            sent_1_chids.discard(max_val[0])
            sent_2_chids.discard(max_val[1])

            if not len(chids_1_used) and not len(chids_2_used):
                alignment = ([max_val[0]], [max_val[1]])
                pair.alignments.append(alignment)
                print_buf.append(get_printable_alignment(pair, alignment, max_val))
            elif len(chids_1_used) and len(chids_2_used):
                continue
            else:
                idx = 0 if len(chids_1_used) else 1
                alignment = next(ali for ali in pair.alignments if
                                 (chids_1_used[0] in ali[0] if idx == 0 else chids_2_used[0] in ali[1]))

                temp_ali = [sub.copy() for sub in alignment]
                temp_ali[idx].append(max_val[idx])

                prev_max_val = (-1, -1,
                                cosine_similarity(get_chunk_vector(vectors, pair.sent_1.chunks_to_words(alignment[0])),
                                                  get_chunk_vector(vectors, pair.sent_2.chunks_to_words(alignment[1]))))
                new_max_val = (max_val[0], max_val[1],
                               cosine_similarity(get_chunk_vector(vectors, pair.sent_1.chunks_to_words(temp_ali[0])),
                                                 get_chunk_vector(vectors, pair.sent_2.chunks_to_words(temp_ali[1]))))

                # OFFSET
                if new_max_val[2] <= prev_max_val[2]:
                    continue

                should_print = True
                alignment[idx].append(max_val[idx])
                alignment[idx].sort()
                print_buf.append(get_printable_alignment(pair, alignment, new_max_val))

        # fun fact vvv doesn't matter
        for chid in sent_1_chids:
            pair.alignments.append(([chid], []))
        for chid in sent_2_chids:
            pair.alignments.append(([], [chid]))

        print_buf.append(pair.alignments)

        if should_print:
            print('\n'.join([(el if type(el) is str else el.__str__()) for el in print_buf]))

    export_and_eval(data)

    print(unrecognized_words)
    print(f'Unrecognized words: {len(unrecognized_words)}')

    # print(lemmastory)


if __name__ == '__main__':
    main()
