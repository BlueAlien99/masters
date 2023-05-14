import concurrent.futures
from helpers.data_exporter import export_and_eval
from helpers.data_reader import get_train_data_gs, get_test_data_gs, get_data_gs
from helpers.data_types import Datasets
from helpers.sentence_pair import SentencePair, Alignment
# from chunker import get_data_gs
from utils.dictionaries import ignore_list, uk_to_us, autocorrect
from utils.math import cosine_similarity
import re
import numpy as np
from typing import Tuple, Callable
import random
from models.keyed_vectors import load_keyed_vectors
from models.tfidf import load_tfidf_model
import matplotlib.pyplot as plt
from statistics import fmean
from bert import chunk_cosine_sim
import torch
import time


# def chunk_cosine_sim(_):
#     return [[torch.Tensor([0]) for _ in range(100)] for _ in range(100)]

DataGetter = Callable[[], list[SentencePair]]

unrecognized_words = set()


kv = load_keyed_vectors()
# dct, tfidf = load_tfidf_model(1/6)


def get_word_vector(vectors: dict, word: str) -> list[list[float]]:
    if word not in vectors:
        return []

    kv_vec = vectors[word]
    # idf_val = tfidf[dct.doc2bow([word])]

    return [kv_vec]

    # if len(idf_val) == 0:
    #     # return []
    #     return [kv_vec]

    # return [[v*idf_val[0][1] for v in kv_vec]]


def preprocess_word(vectors: dict, word: str):
    def _try_with(_word: str):
        if _word is None:
            return None
        _base = _word
        # _base = _word.lower()
        # _base = f'/c/en/{_word}'
        return _base if _base in vectors else None

    # if re.fullmatch('.?(\\d+.?)+', word) is not None:
    #     return 'number'

    candidates = [word]
    # candidates = [word, word[:-1], word[:-2]]
    # candidates = [word, uk_to_us.get(word), autocorrect.get(word)]
    return next((_try_with(cand) for cand in candidates if _try_with(cand) is not None), None)


def preprocess_as_word(prev_word: str, word: str):
    should_pop = False

    if word in ['A', 'B', 'C']:
        if prev_word.lower().startswith('bulb'):
            should_pop = True
        return should_pop, ['bulb', word]

    if word in ['X', 'Y', 'Z']:
        if prev_word.lower().startswith('switch'):
            should_pop = True
        return should_pop, ['switch', word]

    if word.isdigit():
        if prev_word.lower().startswith('terminal'):
            should_pop = True
        return should_pop, ['terminal', word]

    if word.lower().startswith('path'):
        if prev_word.lower() not in ['closed', 'open']:
            return should_pop, ['closed', word]


def preprocess_words(vectors: dict, words: list[str], *, is_as=False):
    if not is_as:
        return words

    new_words: list[str] = []
    for word in words:
        if is_as:
            prev_word = new_words[-1] if len(new_words) else ''
            result = preprocess_as_word(prev_word, word)
            if result is not None:
                if result[0]:
                    new_words.pop()
                new_words.extend(result[1])
                continue

        # new_words.append(word)

        # ^^^ comment when bert VVV

        preprocessed = preprocess_word(vectors, word)
        if preprocessed is not None:
            new_words.append(preprocessed)
            continue

        # word = autocorrect[word] if word in autocorrect else word

        # Handle case like 'non-bathingsuit'
        word_parts = [word_part for word_part in re.split('[^A-Za-z]', word) if word_part]

        for word_part in word_parts:
            preprocessed = preprocess_word(vectors, word_part)
            if preprocessed is not None:
                new_words.append(preprocessed)
            # elif word_part not in ignore_list:
            else:
                # unrecognized_words.add(word_part)
                unrecognized_words.add(word)

    return new_words


def get_chunk_vector(vectors: dict, words: list[str]) -> list[float] | None:
    vecs = [vec for word in words for vec in get_word_vector(vectors, word)]
    return np.mean(vecs, axis=0) if len(vecs) else None


def get_printable_alignment(pair: SentencePair, alignment: Alignment, max_val: Tuple):
    return f'{max_val} => {" ".join(pair.sent_1.chunks_to_words(alignment[0]))} <=> {" ".join(pair.sent_2.chunks_to_words(alignment[1]))}'


def run(data: list[SentencePair], thr: float, *, log=False):
    # print(f'Start for THR = {thr}')
    start_time = time.process_time_ns()

    for pair in data:
        # print(f'Start for THR = {thr}, Pair = {pair.id}')
        print_buf = []

        print_buf.append(f'\n\n>>> PAIR #{pair.id[2] + 1}')
        print_buf.append(' <> '.join([' '.join(cd['words']) for cd in pair.sent_1.chunk_data]))
        print_buf.append(' <> '.join([' '.join(cd['words']) for cd in pair.sent_2.chunk_data]))
        print_buf.append('')

        for sent in [pair.sent_1, pair.sent_2]:
            for chunk in sent.chunk_data:
                chunk['words'] = preprocess_words(kv, chunk['words'], is_as=pair.id[1] == Datasets.AS)

        sim_strings = []

        sims = [[(-1, -1, -1) for _ in range(len(pair.sent_2.chunks))]
                for _ in range(len(pair.sent_1.chunks))]

        for sent in [pair.sent_1, pair.sent_2]:
            for chunk in sent.chunk_data:
                chunk['vec'] = get_chunk_vector(kv, chunk['words'])

        # vvv vector composition
        # for i, chunk_1 in enumerate(pair.sent_1.chunk_data):
        #     for j, chunk_2 in enumerate(pair.sent_2.chunk_data):
        #         sim = cosine_similarity(chunk_1["vec"], chunk_2["vec"])

        #         if len(chunk_1['words']) and ' '.join(chunk_1['words']).lower() == ' '.join(chunk_2['words']).lower():
        #             sim = 1

        #         sims[i][j] = (i, j, sim)
        #         sim_strings.append(
        #             f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')
        # ^^^ vector composition

        # vvv transformsers
        # print(' | '.join([' '.join(data['words']) for data in pair.sent_1.chunk_data]))
        # print(' | '.join([' '.join(data['words']) for data in pair.sent_2.chunk_data]))
        similarities = chunk_cosine_sim(pair)

        skip = [0, 0]
        for i, chunk_1 in enumerate(pair.sent_1.chunk_data):
            if len(chunk_1['words']) == 0:
                skip[0] += 1
                continue

            skip[1] = 0
            for j, chunk_2 in enumerate(pair.sent_2.chunk_data):
                if len(chunk_2['words']) == 0:
                    skip[1] += 1
                    continue

                w2v_sim = cosine_similarity(chunk_1['vec'], chunk_2['vec'])
                tra_sim = similarities[i - skip[0]][j - skip[1]].item()

                # sim = tra_sim
                # sim = (tra_sim + w2v_sim) / 2
                sim = max(tra_sim, w2v_sim)

                # if ' '.join(chunk_1['words']).lower() == ' '.join(chunk_2['words']).lower():
                #     sim = 1

                sims[i][j] = (i, j, sim)
                sim_strings.append(
                    f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')
        # ^^^ transformsers

        # vvv lexical semantic vectors
        # for i, chunk_1 in enumerate(pair.sent_1.chunk_data):
        #     for j, chunk_2 in enumerate(pair.sent_2.chunk_data):
        #         words = {*chunk_1["words"], *chunk_2["words"]}
        #         word_vecs = [vec for word in words for vec in get_word_vector(kv, word)]
        #         chunk_1_word_vecs = [vec for word in chunk_1["words"] for vec in get_word_vector(kv, word)]
        #         chunk_2_word_vecs = [vec for word in chunk_2["words"] for vec in get_word_vector(kv, word)]

        #         chunk_1_vec = []
        #         chunk_2_vec = []
        #         for word_vec in word_vecs:
        #             chunk_1_vec.append(max([cosine_similarity(word_vec, chunk_1_word_vec) for chunk_1_word_vec in
        #                                     chunk_1_word_vecs] or [0]))
        #             chunk_2_vec.append(max([cosine_similarity(word_vec, chunk_2_word_vec) for chunk_2_word_vec in
        #                                     chunk_2_word_vecs] or [0]))

        #         sim = cosine_similarity(chunk_1_vec, chunk_2_vec)
        #         sims[i][j] = (i, j, sim)
        #         sim_strings.append(
        #             f'{"{:.2f}".format(sim)} => {" ".join(chunk_1["words"])} <=> {" ".join(chunk_2["words"])}')
        # ^^^ lexical semantic vectors

        sim_strings.sort(reverse=True)
        print_buf.append('\n'.join([*sim_strings, '']))

        sent_1_chids = set([i for i in range(len(pair.sent_1.chunks))])
        sent_2_chids = set([i for i in range(len(pair.sent_2.chunks))])

        should_print = False

        while True:
            max_val = (-1, -1, -1)
            for i in range(len(sims)):
                for j in range(len(sims[0])):
                    if sims[i][j][2] > max_val[2]:
                        max_val = sims[i][j]
            if max_val[2] < thr:
                break

            chids_1_used = []
            chids_2_used = []
            for ali in pair.alignments:
                if max_val[0] in ali[0] or max_val[1] in ali[1]:
                    chids_1_used.extend([chid for chid in ali[0] if chid != max_val[0]])
                    chids_2_used.extend([chid for chid in ali[1] if chid != max_val[1]])

            sims[max_val[0]][max_val[1]] = (max_val[0], max_val[1], -1)

            if not len(chids_1_used) and not len(chids_2_used):
                alignment = ([max_val[0]], [max_val[1]])
                pair.alignments.append(alignment)
                sent_1_chids.discard(max_val[0])
                sent_2_chids.discard(max_val[1])
                print_buf.append(get_printable_alignment(pair, alignment, max_val))
            elif len(chids_1_used) and len(chids_2_used):
                continue
            elif len(chids_1_used) > 1 or len(chids_2_used) > 1:
                continue
            else:
                idx = 0 if len(chids_1_used) else 1
                alignment = next(ali for ali in pair.alignments if
                                 (chids_1_used[0] in ali[0] if idx == 0 else chids_2_used[0] in ali[1]))

                temp_ali = [sub.copy() for sub in alignment]
                temp_ali[idx].append(max_val[idx])

                prev_max_val = (-1, -1,
                                cosine_similarity(get_chunk_vector(kv, pair.sent_1.chunks_to_words(alignment[0])),
                                                  get_chunk_vector(kv, pair.sent_2.chunks_to_words(alignment[1]))))
                new_max_val = (max_val[0], max_val[1],
                               cosine_similarity(get_chunk_vector(kv, pair.sent_1.chunks_to_words(temp_ali[0])),
                                                 get_chunk_vector(kv, pair.sent_2.chunks_to_words(temp_ali[1]))))

                if new_max_val[2] <= prev_max_val[2]:
                    continue

                should_print = True
                alignment[idx].append(max_val[idx])

                [sent_1_chids, sent_2_chids][idx].discard(max_val[idx])

                alignment[idx].sort()
                print_buf.append(get_printable_alignment(pair, alignment, new_max_val))

        # fun fact vvv doesn't matter
        for chid in sent_1_chids:
            pair.alignments.append(([chid], []))
        for chid in sent_2_chids:
            pair.alignments.append(([], [chid]))

        print_buf.append(pair.alignments)

        if should_print and log:
            print('\n'.join([(el if type(el) is str else el.__str__()) for el in print_buf]))

    dur = time.process_time_ns() - start_time
    dur_ms = round(dur / 1_000_000, 2)
    dur_per_pair = round(dur_ms / len(data), 2)
    # print(f'Aligning -> {dur_ms} ms / {len(data)} pairs = {dur_per_pair} ms / pair')

    return export_and_eval(data, log=log)


def run_train(get_data: DataGetter, thr_step=0.01):
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
        runs = {}

        thr = 0
        while thr <= 1:
            data = get_data()
            r = executor.submit(run, data, thr)
            runs[r] = thr
            thr += thr_step

        for future in concurrent.futures.as_completed(runs):
            thr = runs[future]
            result = future.result()
            results.append((thr, result))

    results.sort(key=lambda r: r[0])
    # print(results)
    plt.plot([d[0] for d in results], [d[1] for d in results])
    plt.axis([0, 1, 0, 1])
    plt.grid()
    # plt.show()
    result = max(results, key=lambda r: r[1])
    return result


def run_test(get_data: DataGetter, thr: float):
    data = get_data()
    result = run(data, thr, log=False)
    # result = run(data, thr, log=True)
    print(f'TEST THR = {thr}')
    print(f'TEST RES = {result}')
    return result


def train_and_test(get_train_data: DataGetter, get_test_data: DataGetter):
    train_result = run_train(get_train_data)
    test_result = run_test(get_test_data, train_result[0])
    run_test(get_test_data, 0)
    return train_result[0], test_result


def main():
    random.seed(a=16032023)

    partial_results = []

    thr = 0.56  # max
    # thr = 0.46  # avg
    # thr = 0.36  # vec comp
    # thr = 0

    print('\nHeadlines:')
    # r = train_and_test(lambda: get_data_gs('train', Datasets.H), lambda: get_data_gs('test', Datasets.H))
    # partial_results.append(r[1])
    run_test(lambda: get_data_gs('test', Datasets.H), thr)

    print('\nImages:')
    # r = train_and_test(lambda: get_data_gs('train', Datasets.I), lambda: get_data_gs('test', Datasets.I))
    # partial_results.append(r[1])
    run_test(lambda: get_data_gs('test', Datasets.I), thr)

    print('\nAnswers-Students:')
    # r = train_and_test(lambda: get_data_gs('train', Datasets.AS), lambda: get_data_gs('test', Datasets.AS))
    # partial_results.append(r[1])
    run_test(lambda: get_data_gs('test', Datasets.AS), thr)

    # print(f'\nAvg: {fmean(partial_results)}')

    # print('\nAll:')
    # train_and_test(get_train_data_gs, get_test_data_gs)

    # print(f'Number of unrecognized words: {len(unrecognized_words)}')
    # print(unrecognized_words)


if __name__ == '__main__':
    main()
