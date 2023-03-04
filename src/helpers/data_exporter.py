import os

from .data_types import Datasets, DataType
from .sentence_pair import SentencePair

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_export_path(datatype: DataType, dataset: Datasets):
    return f'{dir_path}/../../output/STSint.{"test" if datatype == "test" else ""}output.{dataset}.wa'


def export_data_to_wa_files(data: list[SentencePair]):
    datasets = {}
    for pair in data:
        dataset = pair.id[:-1]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(pair)

    for dataset in datasets:
        data = datasets[dataset]
        data.sort(key=lambda p: p.id)
        with open(get_export_path(*dataset), 'w') as f:
            f.write('\n\n\n'.join([pair.to_wa_entry_string() for pair in data]))
