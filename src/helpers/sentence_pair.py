from dataclasses import dataclass
from typing import Tuple

from .data_types import Datasets, DataType
from .sentence import Sentence

Alignment = Tuple[list[int], list[int]]
SPID = Tuple[DataType, Datasets, int]

NOALI = '-not aligned-'


def tokens_to_wa_token_string(tokens: list[int]):
    return ' '.join([str(token + 1) for token in (tokens or [-1])])


@dataclass
class SentencePair:
    id: SPID
    sent_1: Sentence
    sent_2: Sentence
    alignments: list[Alignment]

    def __init__(self, spid: SPID, sent_1: Sentence, sent_2: Sentence):
        self.id = spid
        self.sent_1 = sent_1
        self.sent_2 = sent_2
        self.alignments = []

    def to_wa_alignment_strings(self):
        alignment_strings = []
        for ali in self.alignments:
            sent_1_tokens = self.sent_1.chunks_to_tokens(ali[0])
            sent_2_tokens = self.sent_2.chunks_to_tokens(ali[1])
            is_noali = len(sent_1_tokens) == 0 or len(sent_2_tokens) == 0

            alignment_strings.append(''.join([
                tokens_to_wa_token_string(sent_1_tokens),
                ' <==> ',
                tokens_to_wa_token_string(sent_2_tokens),
                f' // {"NOALI" if is_noali else "EQUI"}',
                f' // {"NIL" if is_noali else "5"}',
                f' // {self.sent_1.tokens_to_string(sent_1_tokens) or NOALI}',
                ' <==> ',
                f'{self.sent_2.tokens_to_string(sent_2_tokens) or NOALI} ',
            ]))

        return alignment_strings

    def to_wa_entry_string(self):
        return '\n'.join([
            f'<sentence id="{self.id[2] + 1}" status="">',
            f'// {self.sent_1.string}',
            f'// {self.sent_2.string}',
            '<source>',
            *self.sent_1.to_wa_token_mappings(),
            '</source>',
            '<translation>',
            *self.sent_2.to_wa_token_mappings(),
            '</translation>',
            '<alignment>',
            *self.to_wa_alignment_strings(),
            '</alignment>',
            '</sentence>',
        ])
