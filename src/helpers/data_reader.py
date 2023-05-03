import re

from . import paths
from .data_types import Datasets, DataType
from .sentence import Sentence
from .sentence_pair import SentencePair


def _get_data(datatype: DataType, dataset: Datasets, chunked: bool):
    filenames = [
        f'STSint.{"test" if datatype == "test" else ""}input.{dataset}.sent{i}{".chunk" if chunked else ""}.txt' for i
        in [1, 2]]
    filepaths = [f'{paths.data_path}/{filename}' for filename in filenames]

    files_contents = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            file_content = [sentence.strip() for sentence in f.readlines()]
            files_contents.append(file_content)

    return list(zip(*files_contents))


def get_data_gs_with_ali(datatype: DataType, dataset: Datasets):
    NOALI = '-not aligned-'

    data = get_data_gs(datatype, dataset)

    filename = f'STSint.{"test" if datatype == "test" else ""}input.{dataset}.wa'
    filepath = f'{paths.data_path}/{filename}'

    pairs: list[str] = []
    with open(filepath, 'r') as f:
        pairs = re.findall(re.compile(r'<alignment>(.*?)</alignment>', re.S), f.read())

    alis = [[[a.strip() for a in ali.split('//')[-1].split('<==>')]
             for ali in pair.strip().splitlines()] for pair in pairs]

    skipped = set()

    for pair, ali in zip(data, alis):
        sent_1_chunks = [pair.sent_1.tokens_to_string(chunk) for chunk in pair.sent_1.chunks]
        sent_2_chunks = [pair.sent_2.tokens_to_string(chunk) for chunk in pair.sent_2.chunks]

        def decode(ali_str: str, chunks: list[str]):
            chunks_sorted = [*chunks]
            chunks_sorted.sort(key=len, reverse=True)

            chids = []
            while ali_str and ali_str != NOALI:
                did_break = False
                for c in chunks_sorted:
                    if ali_str.startswith(c):
                        ali_str = ali_str[len(c):].strip()
                        chids.append(chunks.index(c))
                        did_break = True
                        break
                if not did_break:
                    skipped.add(pair.id[2] + 1)
                    break
            return chids

        for a_1, a_2, in ali:
            chid_ali = (decode(a_1, sent_1_chunks),
                        decode(a_2, sent_2_chunks))
            pair.alignments.append(chid_ali)

    print(f'Skipped at least one alignment in: {skipped}')

    return data


def get_data_gs(datatype: DataType, dataset: Datasets):
    data = _get_data(datatype, dataset, True)
    data = [list(map(Sentence.from_chunks, pair)) for pair in data]
    data = [SentencePair((datatype, dataset, i), *pair) for (i, pair) in enumerate(data)]

    return data


def get_data_sys(datatype: DataType, dataset: Datasets):
    data = _get_data(datatype, dataset, False)
    data = [list(map(Sentence.from_sentence, pair)) for pair in data]
    data = [SentencePair((datatype, dataset, i), *pair) for (i, pair) in enumerate(data)]

    return data


def get_train_data_gs():
    return [*get_data_gs('train', Datasets.H), *get_data_gs('train', Datasets.I), *get_data_gs('train', Datasets.AS)]


def get_test_data_gs():
    return [*get_data_gs('test', Datasets.H), *get_data_gs('test', Datasets.I), *get_data_gs('test', Datasets.AS)]


def get_all_data_gs():
    return [*get_train_data_gs(), *get_test_data_gs()]
