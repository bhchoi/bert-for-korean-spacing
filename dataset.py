from preprocessor import Preprocessor
from torch.utils.data import Dataset
from utils import load_slot_labels


class SpacingDataset(Dataset):
    def __init__(self, data_path: str, preprocessor: Preprocessor):
        self.sentences = []
        self.preprocessor = preprocessor
        self.slot_labels = load_slot_labels()

        self.__load_data(data_path)

    def __load_data(self, data_path):
        with open(data_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            self.sentences = [line.split() for line in lines]

    def __transform(self, sentence):
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
        tags = self.__transform(self.sentences[idx])
        tags = [self.slot_labels.index(t) for t in tags]

        (
            input_ids,
            slot_labels,
            attention_mask,
            token_type_ids,
        ) = self.preprocessor.get_input_features(sentence, tags)

        return input_ids, slot_labels, attention_mask, token_type_ids
