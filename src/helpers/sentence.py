import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Sentence:
    string: str
    tokens: list[str]
    chunks: Optional[list[list[int]]]
    chunk_data: list[dict[str, any]]

    def __init__(self, sentence: str, tokens: list[str], chunks: Optional[list[list[int]]] = None):
        self.string = sentence
        self.tokens = tokens
        self.chunks = chunks
        self.chunk_data = [{'words': self.tokens_to_string(chunk).split(' ')} for chunk in chunks]

    @staticmethod
    def from_sentence(sentence: str):
        return Sentence(sentence, sentence.split())

    @staticmethod
    def from_chunks(chunks_str: str):
        tokens = [token for token in re.split('[ \\[\\]]+', chunks_str) if token]
        sentence = ' '.join(tokens)
        chunks = [chunk.strip() for chunk in re.findall('(?<=\\[)[^\\[\\]]+', chunks_str)]
        chunks_id = [[tokens.index(token) for token in chunk.split()] for chunk in chunks]
        return Sentence(sentence, tokens, chunks_id)

    def to_wa_token_mappings(self):
        return [f'{i + 1} {token} : ' for (i, token) in enumerate(self.tokens)]

    def chunks_to_tokens(self, chunks: list[int]):
        return [token for chunk in chunks for token in self.chunks[chunk]]

    def tokens_to_string(self, tokens: list[int]):
        return ' '.join([self.tokens[token] for token in tokens])
