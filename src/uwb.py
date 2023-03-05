from helpers.data_exporter import export_and_eval
from helpers.data_reader import get_data_gs
from helpers.data_types import Datasets


def process_sentence(pair):
    pass


def main():
    data = get_data_gs('test', Datasets.H)

    for pair in data:
        ali = min(len(pair.sent_1.chunks), len(pair.sent_2.chunks))
        no_ali = max(len(pair.sent_1.chunks), len(pair.sent_2.chunks))
        first_longer = len(pair.sent_1.chunks) > len(pair.sent_2.chunks)
        pair.alignments = [*[([i], [i]) for i in range(ali)],
                           *[(([i], []) if first_longer else ([], [i])) for i in range(ali, no_ali)]]

    export_and_eval(data)


if __name__ == '__main__':
    main()
