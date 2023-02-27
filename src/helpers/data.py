import os
from enum import Enum
from typing import Literal

dir_path = os.path.dirname(os.path.realpath(__file__))


class Datasets(str, Enum):
    AS = 'answers-students'
    H = 'headlines'
    I = 'images'


DataType = Literal['test', 'train']


def _get_data(datatype: DataType, dataset: Datasets, chunked: bool):
    filenames = [
        f'STSint.{"test" if datatype == "test" else ""}input.{dataset}.sent{i}{".chunk" if chunked else ""}.txt' for i
        in [1, 2]]
    filepaths = list(map(lambda filename: f'{dir_path}/../../data/{filename}', filenames))

    files_contents = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            file_content = list(map(lambda sent: sent.strip(), f.readlines()))
            files_contents.append(file_content)

    return list(zip(*files_contents))


def get_data_gs(datatype: DataType, dataset: Datasets):
    data = _get_data(datatype, dataset, True)
    data = list(map(lambda pair: list(
        map(lambda sent: list(map(lambda chunk: chunk.strip(), sent.strip()[1:-1].split('] ['))), pair)), data))

    return data


def get_data_sys(datatype: DataType, dataset: Datasets):
    return _get_data(datatype, dataset, False)
