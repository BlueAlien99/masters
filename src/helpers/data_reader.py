import os

from .data_types import Datasets, DataType
from .sentence import Sentence
from .sentence_pair import SentencePair

dir_path = os.path.dirname(os.path.realpath(__file__))


def _get_data(datatype: DataType, dataset: Datasets, chunked: bool):
    filenames = [
        f'STSint.{"test" if datatype == "test" else ""}input.{dataset}.sent{i}{".chunk" if chunked else ""}.txt' for i
        in [1, 2]]
    filepaths = [f'{dir_path}/../../data/{filename}' for filename in filenames]

    files_contents = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            file_content = [sentence.strip() for sentence in f.readlines()]
            files_contents.append(file_content)

    return list(zip(*files_contents))


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
