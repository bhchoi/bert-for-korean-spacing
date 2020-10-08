import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AdamW
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score

from utils import load_slot_labels


class SpacingBertModel(pl.LightningModule):
    def __init__(
        self,
        args: argparse,
        ner_train_dataloader: DataLoader,
        ner_val_dataloader: DataLoader,
        ner_test_dataloader: DataLoader,
    ):
        super().__init__()
        self.args = args
        self.ner_train_dataloader = ner_train_dataloader
        self.ner_val_dataloader = ner_val_dataloader
        self.ner_test_dataloader = ner_test_dataloader
        self.slot_labels_type = load_slot_labels()
        self.ignore_index = torch.nn.CrossEntropyLoss().ignore_index

        self.config = BertConfig.from_pretrained(
            self.args.bert_model, num_labels=len(self.slot_labels_type)
        )
        # self.model = BertModel.from_pretrained(self.args.bert_model, config=self.config)
        self.model = BertModel.from_pretrained(self.args.bert_model)
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.linear = nn.Linear(self.config.hidden_size, len(load_slot_labels()))

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        x = outputs[0]
        x = self.dropout(x)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_labels)
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = self._calculate_loss(outputs, slot_labels)
        gt_slot_labels, pred_slot_labels = self._convert_ids_to_labels(
            outputs, slot_labels
        )

        val_acc = self._f1_score(gt_slot_labels, pred_slot_labels)

        return {"val_loss": loss, "val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        tensorboard_log = {
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        return {"val_loss": val_loss, "progress_bar": tensorboard_log}

    def test_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, slot_labels = batch

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        gt_slot_labels, pred_slot_labels = self._convert_ids_to_labels(
            outputs, slot_labels
        )

        test_acc = self._f1_score(gt_slot_labels, pred_slot_labels)

        test_step_outputs = {
            "test_acc": test_acc,
            "gt_labels": gt_slot_labels,
            "pred_labels": pred_slot_labels,
        }

        return test_step_outputs

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        gt_labels = []
        pred_labels = []
        for x in outputs:
            gt_labels.extend(x["gt_labels"])
            pred_labels.extend(x["pred_labels"])

        test_step_outputs = {
            "test_acc": test_acc,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
        }

        return test_step_outputs

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.ner_train_dataloader

    def val_dataloader(self):
        return self.ner_val_dataloader

    def test_dataloader(self):
        return self.ner_test_dataloader

    def _calculate_loss(self, outputs, labels):
        # active_loss = attention_mask.view(-1) == 1
        # active_logits = outputs.view(-1, len(self.slot_labels_type))[active_loss]
        # active_labels = slot_labels.view(-1)[active_loss]
        active_logits = outputs.view(-1, len(self.slot_labels_type))
        active_labels = labels.view(-1)
        loss = F.cross_entropy(active_logits, active_labels)

        return loss

    def _f1_score(self, gt_slot_labels, pred_slot_labels):
        return torch.tensor(
            f1_score(gt_slot_labels, pred_slot_labels), dtype=torch.float32
        )

    def _convert_ids_to_labels(self, outputs, slot_labels):
        _, y_hat = torch.max(outputs, dim=2)
        y_hat = y_hat.detach().cpu().numpy()
        slot_label_ids = slot_labels.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.slot_labels_type)}
        slot_gt_labels = [[] for _ in range(slot_label_ids.shape[0])]
        slot_pred_labels = [[] for _ in range(slot_label_ids.shape[0])]

        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != self.ignore_index:
                    slot_gt_labels[i].append(slot_label_map[slot_label_ids[i][j]])
                    slot_pred_labels[i].append(slot_label_map[y_hat[i][j]])

        return slot_gt_labels, slot_pred_labels