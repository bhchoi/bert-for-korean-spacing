import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import SpacingDataset
from net import SpacingBertModel


def get_dataloader(data_path, preprocessor, batch_size):
    dataset = SpacingDataset(data_path, preprocessor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


def main(config):

    preprocessor = Preprocessor(config.bert_model, config.max_len)
    test_dataloader = get_dataloader(
        config.test_data_path, preprocessor, config.eval_batch_size
    )
    model = SpacingBertModel(config, None, None, test_dataloader)
    checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer()
    res = trainer.test(model)


if __name__ == "__main__":
    config = OmegaConf.load("config/eval_config.yaml")
    main(config)