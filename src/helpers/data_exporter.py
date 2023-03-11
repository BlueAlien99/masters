import re
from datetime import datetime
from statistics import fmean
from subprocess import check_output

from . import paths
from .data_types import Datasets, DataType
from .sentence_pair import SentencePair


def get_export_path(datatype: DataType, dataset: Datasets):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    return f'{paths.output_path}/{timestamp}-STSint.{"test" if datatype == "test" else ""}output.{dataset}.wa'


def export_data_to_wa_files(data: list[SentencePair]):
    datasets = {}
    export_paths = []

    for pair in data:
        dataset = pair.id[:-1]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(pair)

    for dataset in datasets:
        data = datasets[dataset]
        data.sort(key=lambda p: p.id)
        export_path = get_export_path(*dataset)
        export_paths.append(export_path)
        with open(export_path, 'w') as f:
            f.write('\n\n\n'.join([pair.to_wa_entry_string() for pair in data]))

    return export_paths


def perl_eval(input_path: str, export_path: str, label=''):
    cmd = f'perl {paths.eval_script} {input_path} {export_path}'
    output = check_output(cmd.split()).decode()
    scores = [float(re.search('\\d\\.\\d+', line).group()) for line in output.splitlines()]

    print(f'\n{label}')
    print(export_path.split('/')[-1])
    print(output)

    return scores


def export_and_eval(data: list[SentencePair], label=''):
    export_paths = export_data_to_wa_files(data)
    ali_scores = []
    for path in export_paths:
        filename = path.split('/')[-1]
        input_path = f'{paths.data_path}/{filename.split("-", 1)[-1].replace("output", "input")}'
        scores = perl_eval(input_path, path, label)
        ali_scores.append(scores[0])
    avg_ali_score = round(fmean(ali_scores), 4)
    print(f'\n F1 Ali Avg {avg_ali_score}\n\n')
