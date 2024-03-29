import concurrent.futures
import nltk
from nltk.chunk import regexp as NCRE
from helpers.data_reader import get_data_sys, get_data_gs as _get_data_gs
from helpers.data_types import DataType, Datasets
from helpers.sentence_pair import SentencePair
import helpers.paths as paths
from helpers.sentence import Sentence
import spacy
import time

# nltk.download('averaged_perceptron_tagger')
sp = spacy.load('en_core_web_trf')
# sp = spacy.load('en_core_web_sm')


POS_MAP = {
    'closed': 'JJ'
}


def export_chunks_diff(gs_data: list[SentencePair], chunker_data: list[SentencePair]):
    # timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    timestamp = 'now'
    export_path = f'{paths.output_path}/{timestamp}-chunks.diff'

    with open(export_path, 'w') as f:
        for gs, ch in zip(gs_data, chunker_data):
            for gs_sent, ch_sent in zip([gs.sent_1, gs.sent_2], [ch.sent_1, ch.sent_2]):
                gs_ali = gs_sent.to_chunks_str()
                ch_ali = ch_sent.to_chunks_str()

                if gs_ali != ch_ali:
                    f.write(f'\n{gs_ali}\n{ch_ali}\n')


def get_data_gs(datatype: DataType, dataset: Datasets):
    data = get_data_sys(datatype, dataset)
    total_pos_duration = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        sents = {}

        for pair in data:
            for i, sent in enumerate([pair.sent_1, pair.sent_2]):
                r = executor.submit(chunk_sentence, sent.string)
                sents[r] = (*pair.id, i)

        for future in concurrent.futures.as_completed(sents):
            sent_id = sents[future]
            chunks, dur = future.result()
            pair = next(p for p in data if p.id == sent_id[:-1])

            total_pos_duration += dur

            if sent_id[-1] == 0:
                pair.sent_1 = Sentence.from_chunks(chunks)
            else:
                pair.sent_2 = Sentence.from_chunks(chunks)

    n_sent = len(data) * 2
    pos_dur_ms = round(total_pos_duration / 1_000_000, 2)
    dur_per_sent = round(pos_dur_ms / n_sent, 2)
    print(f'POS taging -> {pos_dur_ms} ms / {n_sent} sents = {dur_per_sent} ms / sent')

    return data


def chunk_sentence(sent: str) -> tuple[str, int]:
    init_tokens = sent.split()

    start_time = time.process_time_ns()

    doc = sp(sent)
    # raw_tags = [(token.text, token.tag_) for token in doc]
    raw_tags = [(token.text, POS_MAP[token.text] if token.text in POS_MAP else token.tag_) for token in doc]

    tags = []
    curr_init_token = 0
    is_full_token = True

    for tag in raw_tags:
        if is_full_token:
            tags.append(tag)
        else:
            tags[-1] = (f'{tags[-1][0]}{tag[0]}', tags[-1][1])

        is_full_token = False
        init_tokens[curr_init_token] = init_tokens[curr_init_token][len(tag[0]):]

        if len(init_tokens[curr_init_token]) == 0:
            is_full_token = True
            curr_init_token += 1

    # OR
    # tags = nltk.pos_tag(sent.split())

    pos_duration = time.process_time_ns() - start_time

    # print(tags)

    rules = [
        NCRE.ChunkRule(r"<.>", 'periods'),
        NCRE.ChunkRule(r"<WP>", 'who, what, how, etc.'),
        NCRE.ChunkRule(r"<EX>", 'there'),
        NCRE.ChunkRule(r"<TO>?(<RB>|<MD>|<VB.*>)*<VB.*>(<RB>|<RP>)?", 'verb chain'),
        NCRE.ChunkRule(r"<POS>?(<TO>|<IN>)?((<DT>|<PRP\$>)?<CD>?<RB>?<JJ.*>*<NN.*>*|<CC>)*<NN.*><CD>?", 'NP / PP'),
        NCRE.ChunkRule(r"<.*>+", 'Create missing chunks'),
        NCRE.MergeRule(r"<CD>", r"<CD>?<CC><CD>", '(terminals) 1 + (2) and 3'),
    ]

    chunk_parser = NCRE.RegexpChunkParser(rules, chunk_label='NP')
    chunked_text = chunk_parser.parse(tags)
    chunked_sent = ''

    for n in chunked_text:
        if isinstance(n, nltk.tree.Tree):
            if n.label() == 'NP':
                real_str = ' '.join([tag.rsplit('/', 1)[0] for tag in str(n).split() if '/' in tag])
                chunked_sent += f'[ {real_str} ]'

    return chunked_sent, pos_duration


def main():
    # export_chunks_diff(_get_data_gs('test', Datasets.H), get_data_gs('test', Datasets.H))
    # export_chunks_diff(_get_data_gs('test', Datasets.I), get_data_gs('test', Datasets.I))
    # export_chunks_diff(_get_data_gs('test', Datasets.AS), get_data_gs('test', Datasets.AS))

    all_gs = [*_get_data_gs('test', Datasets.H), *_get_data_gs('test', Datasets.I), *_get_data_gs('test', Datasets.AS)]
    all_ch = [*get_data_gs('test', Datasets.H), *get_data_gs('test', Datasets.I), *get_data_gs('test', Datasets.AS)]

    export_chunks_diff(all_gs, all_ch)


if __name__ == '__main__':
    main()
