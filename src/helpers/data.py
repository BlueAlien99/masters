import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

dir_path = os.path.dirname(os.path.realpath(__file__))


class Datasets(str, Enum):
    AS = 'answers-students'
    H = 'headlines'
    I = 'images'


DataType = Literal['test', 'train']


@dataclass
class Sentence:
    string: str
    tokens: list[str]
    chunks: Optional[list[list[int]]]

    def __init__(self, sentence: str, tokens: list[str], chunks: Optional[list[list[int]]] = None):
        self.string = sentence
        self.tokens = tokens
        self.chunks = chunks

    @staticmethod
    def from_sentence(sentence: str):
        return Sentence(sentence, sentence.split())

    @staticmethod
    def from_chunks(chunks_str: str):
        chunks = [chunk.strip()
                  for chunk in chunks_str.strip()[1:-1].split('] [')]
        sentence = ' '.join(chunks)
        tokens = sentence.split()
        chunks_id = [[tokens.index(token)
                      for token in chunk.split()] for chunk in chunks]
        return Sentence(sentence, tokens, chunks_id)


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

    return data


def get_data_sys(datatype: DataType, dataset: Datasets):
    data = _get_data(datatype, dataset, False)
    data = [list(map(Sentence.from_sentence, pair)) for pair in data]

    return data
