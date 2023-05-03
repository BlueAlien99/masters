import matplotlib.pyplot as plt
import matplotlib.colors

from helpers.data_reader import get_data_gs_with_ali
from helpers.data_types import Datasets, DataType
from helpers.data_exporter import export_and_eval


MAX_SIZE = 5
alignment_matrix = [[0 for _ in range(MAX_SIZE)] for _ in range(MAX_SIZE)]


def analyze(datatype: DataType, dataset: Datasets):
    print('\n---- ---- ---- ----\n')

    data = get_data_gs_with_ali(datatype, dataset)
    export_and_eval(data, log=True)

    stats = {
        'noali': 0,
        'one-to-one': 0,
        'one-to-many': 0,
        'many-to-many': 0,
    }
    other = 0

    for pair in data:
        for ali in pair.alignments:
            if len(ali[0]) == 0 or len(ali[1]) == 0:
                stats['noali'] += 1
            elif len(ali[0]) == 1 and len(ali[1]) == 1:
                stats['one-to-one'] += 1
            elif (len(ali[0]) == 1 and len(ali[1]) > 1) or (len(ali[0]) > 1 and len(ali[1]) == 1):
                stats['one-to-many'] += 1
            elif len(ali[0]) > 1 and len(ali[1]) > 1:
                stats['many-to-many'] += 1
            else:
                other += 1

            alignment_matrix[min(len(ali[0]), MAX_SIZE-1)][min(len(ali[1]), MAX_SIZE-1)] += 1

    assert (other == 0)
    print(stats)


def main():
    analyze('train', Datasets.I)
    analyze('train', Datasets.H)
    analyze('train', Datasets.AS)

    analyze('test', Datasets.I)
    analyze('test', Datasets.H)
    analyze('test', Datasets.AS)

    fig, ax = plt.subplots()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['white', (0.1, 0.5, 0.9)])
    ax.matshow(alignment_matrix, cmap=cmap, norm=matplotlib.colors.LogNorm())

    for i in range(MAX_SIZE):
        for j in range(MAX_SIZE):
            value = alignment_matrix[i][j]
            ax.text(j, i, f'{value if value else "-"}', ha='center', va='center')

    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[MAX_SIZE] = f'{MAX_SIZE-1}\nor more'

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.show()


if __name__ == '__main__':
    main()
