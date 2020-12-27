from typing import Callable, List, Tuple
from torch.utils.data import Dataset

from utils import load_slot_labels


class CorpusDataset(Dataset):
    def __init__(self, data_path: str, transform: Callable[[List, List], Tuple]):
        self.sentences = []
        self.transform = transform
        self.slot_labels = load_slot_labels()

        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """data를 file에서 불러온다.

        Args:
            data_path: file 경로
        """
        with open(data_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            self.sentences = [line.split() for line in lines]

    def _get_tags(self, sentence: List[str]) -> List[str]:
        """문장에 대해 띄어쓰기 tagging을 한다.
        character 단위로 분리하여 BIES tagging을 한다.

        Args:
            sentence: 문장

        Retrns:
            문장의 각 토큰에 대해 tagging한 결과 리턴
            ["B", "I", "E"]
        """

        all_tags = []
        for word in sentence:
            if len(word) == 1:
                all_tags.append("S")
            elif len(word) > 1:
                for i, c in enumerate(word):
                    if i == 0:
                        all_tags.append("B")
                    elif i == len(word) - 1:
                        all_tags.append("E")
                    else:
                        all_tags.append("I")
        return all_tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = "".join(self.sentences[idx])
        tags = self._get_tags(self.sentences[idx])
        tags = [self.slot_labels.index(t) for t in tags]

        (
            input_ids,
            slot_labels,
            attention_mask,
            token_type_ids,
        ) = self.transform(sentence, tags)

        return input_ids, slot_labels, attention_mask, token_type_ids
