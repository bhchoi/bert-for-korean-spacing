import torch
from typing import List, Tuple

from tokenization_kobert import KoBertTokenizer


class Preprocessor:
    def __init__(self, max_len: int):
        self.tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
        self.max_len = max_len
        self.pad_token_id = 0

    def get_input_features(
        self, sentence: List[str], tags: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """문장과 띄어쓰기 tagging에 대해 feature로 변환한다.

        Args:
            sentence: 문장
            tags: 띄어쓰기 tagging

        Returns:
            feature를 리턴한다.
            input_ids, attention_mask, token_type_ids, slot_labels
        """

        input_tokens = []
        slot_labels = []

        for word, tag in zip(sentence, tags):
            tokens = self.tokenizer.tokenize(word)

            if len(tokens) == 0:
                tokens = self.tokenizer.unk_token

            input_tokens.extend(tokens)

            for i, _ in enumerate(tokens):
                if i == 0:
                    slot_labels.extend([tag])
                else:
                    slot_labels.extend([self.pad_token_id])

        # 2. max_len보다 길이가 길면 뒤에 자르기
        if len(input_tokens) > self.max_len - 2:
            input_tokens = input_tokens[: self.max_len - 2]
            slot_labels = slot_labels[: self.max_len - 2]

        # cls, sep 추가
        input_tokens = (
            [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
        )
        slot_labels = [self.pad_token_id] + slot_labels + [self.pad_token_id]

        # token을 id로 변환
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # padding
        pad_len = self.max_len - len(input_tokens)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * pad_len)
        slot_labels = slot_labels + ([self.pad_token_id] * pad_len)
        attention_mask = attention_mask + ([0] * pad_len)
        token_type_ids = token_type_ids + ([0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        slot_labels = torch.tensor(slot_labels, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, slot_labels
