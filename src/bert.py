# Based on https://github.com/manishb89/interpretable_sentence_similarity/blob/ebf130d696333eeb3f7613e0c9799f8895dc865c/src/corpus/chunk_embedding.py

import torch
from transformers import AutoTokenizer, AutoModel, logging

from helpers.sentence_pair import SentencePair
from helpers.sentence import Sentence

logging.set_verbosity_error()

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.eval()


def get_bert_sentence_embedding(sentence: str):
    marked_text = "[CLS] " + sentence + " [SEP]"

    token_sub_word_mapping = {}
    tokenized_text = []
    basic_tokens = marked_text.split()

    for token in basic_tokens:
        split_tokens = []
        # for sub_token in bert_tokenizer.wordpiece_tokenizer.tokenize(token):
        for sub_token in bert_tokenizer.tokenize(token):
            split_tokens.append(sub_token)
            tokenized_text.append(sub_token)
        token_sub_word_mapping[token] = split_tokens

    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        output = bert_model(tokens_tensor, segments_tensors)
        encoded_layers = getattr(output, 'last_hidden_state')

        batch_i = 0
        token_embeddings = []

        # For each token in the sentence...
        for token_i in range(len(tokenized_text)):

            # Holds 12 layers of hidden states for each token
            hidden_layers = []

            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                # vec = encoded_layers[layer_i][batch_i][token_i]
                vec = encoded_layers[layer_i][token_i]

                hidden_layers.append(vec)

            token_embeddings.append(hidden_layers)

        sub_word_embeddings = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]

        final_token_embeddings: list[torch.Tensor] = []
        idx_counter = 0

        for token in basic_tokens:
            sub_words = token_sub_word_mapping[token]

            if len(sub_words) == 1:
                final_token_embeddings.append(sub_word_embeddings[idx_counter])
                idx_counter += 1
            else:
                sub_words_emb_list = []
                for i in range(len(sub_words)):
                    sub_words_emb_list.append(sub_word_embeddings[idx_counter + i])
                final_token_embeddings.append(torch.mean(torch.stack(sub_words_emb_list), dim=0))
                idx_counter += len(sub_words)

    return basic_tokens, final_token_embeddings


def get_bert_chunks_embedding(tokenized_text: list[str], sentence_embedding: list[torch.Tensor], sent: Sentence):
    emb_vector_list = []
    for tokens in sent.chunks:
        start_index, end_index = tokens[0], tokens[-1]
        start_token, end_token = sent.tokens[start_index], sent.tokens[end_index]
        start_bert_token, end_bert_token = tokenized_text[start_index+1], tokenized_text[end_index+1]
        if start_token != start_bert_token or end_token != end_bert_token:
            print("Something is wrong somewhere!", tokenized_text, sent.string, tokens)

        start_chunk_embedding = sentence_embedding[start_index+1]
        end_chunk_embedding = sentence_embedding[end_index+1]
        chunk_embedding = torch.cat((start_chunk_embedding, end_chunk_embedding), 0)
        emb_vector_list.append(chunk_embedding)

    return torch.stack(emb_vector_list, dim=0)


def chunk_cosine_sim(pair: SentencePair):
    tokenized_left_text, left_sentence_embedding = get_bert_sentence_embedding(pair.sent_1.string)
    tokenized_right_text, right_sentence_embedding = get_bert_sentence_embedding(pair.sent_2.string)

    left_matrix = get_bert_chunks_embedding(tokenized_left_text, left_sentence_embedding, pair.sent_1)
    right_matrix = get_bert_chunks_embedding(tokenized_right_text, right_sentence_embedding, pair.sent_2)

    left_norm = left_matrix / left_matrix.norm(dim=1)[:, None]
    right_norm = right_matrix / right_matrix.norm(dim=1)[:, None]
    cosine_sim = torch.mm(left_norm, right_norm.transpose(0, 1))

    return cosine_sim
