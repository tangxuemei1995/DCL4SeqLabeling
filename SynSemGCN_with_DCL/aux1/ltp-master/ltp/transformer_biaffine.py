#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import namedtuple
from argparse import ArgumentParser

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from ltp.nn import MLP, Bilinear, BaseModule

GraphResult = namedtuple('GraphResult', ['loss', 'src_arc_logits', 'arc_logits', 'rel_logits'])


def dep_loss(model, s_arc, s_rel, head, labels, mask):
    head_loss = nn.CrossEntropyLoss()
    rel_loss = nn.CrossEntropyLoss()

    # ignore the first token of each sentence
    s_arc = s_arc[:, 1:, :]
    s_rel = s_rel[:, 1:, :]

    # Only keep active parts of the loss
    active_heads = head[mask]
    active_labels = labels[mask]
    s_arc, s_rel = s_arc[mask], s_rel[mask]

    s_rel = s_rel[torch.arange(len(active_heads)), active_heads]

    arc_loss = head_loss(s_arc, active_heads)
    rel_loss = rel_loss(s_rel, active_labels)
    loss = 2 * ((1 - model.loss_interpolation) * arc_loss + model.loss_interpolation * rel_loss)

    return loss


def sdp_loss(model, s_arc, s_rel, head, labels, mask):
    head_loss = nn.BCEWithLogitsLoss()
    rel_loss = nn.CrossEntropyLoss()

    # ignore the first token of each sentence
    s_arc = s_arc[:, 1:, :]
    s_rel = s_rel[:, 1:, :]

    # mask
    mask = mask.unsqueeze(-1).expand_as(s_arc)

    arc_loss = head_loss(s_arc[mask], head[mask].float())
    rel_loss = rel_loss(s_rel[head > 0], labels[head > 0])

    loss = 2 * ((1 - model.loss_interpolation) * arc_loss + model.loss_interpolation * rel_loss)

    return loss


class BiaffineClassifier(nn.Module):
    def __init__(self, input_size, label_num, dropout, arc_hidden_size=500, rel_hidden_size=100,
                 loss_interpolation=0.4, loss_func=dep_loss, char_based=False):
        super().__init__()
        self.char_based = char_based
        self.label_num = label_num
        self.loss_interpolation = loss_interpolation

        self.mlp_arc_h = MLP([input_size, arc_hidden_size], output_dropout=dropout, output_activation=nn.ReLU)
        self.mlp_arc_d = MLP([input_size, arc_hidden_size], output_dropout=dropout, output_activation=nn.ReLU)
        self.mlp_rel_h = MLP([input_size, rel_hidden_size], output_dropout=dropout, output_activation=nn.ReLU)
        self.mlp_rel_d = MLP([input_size, rel_hidden_size], output_dropout=dropout, output_activation=nn.ReLU)

        self.arc_atten = Bilinear(arc_hidden_size, arc_hidden_size, 1, bias_x=True, bias_y=False, expand=True)
        self.rel_atten = Bilinear(rel_hidden_size, rel_hidden_size, label_num, bias_x=True, bias_y=True, expand=True)

        self.loss_func = loss_func

    def forward(self, input, attention_mask=None, word_index=None,
                word_attention_mask=None, head=None, labels=None, is_processed=False):
        if not is_processed:
            input = input[:, :-1, :]
            if self.char_based:
                mask = attention_mask[:, 2:] == 1
                # use bigram ?
                # bigram = torch.cat([input[:, :-1, :].unsqueeze(2), input[:, 1:, :].unsqueeze(2)], dim=2)
                # bigram = torch.mean(bigram, dim=2)
                # input = torch.cat([input[:, :1, :], bigram], dim=1)
            else:
                mask = word_attention_mask
                if word_index is not None:
                    input = torch.cat([input[:, :1, :], torch.gather(
                        input[:, 1:, :], dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, input.size(-1))
                    )], dim=1)
        else:
            mask = word_attention_mask

        arc_h = self.mlp_arc_h(input)
        arc_d = self.mlp_arc_d(input)

        rel_h = self.mlp_rel_h(input)
        rel_d = self.mlp_rel_d(input)

        s_arc = self.arc_atten(arc_d, arc_h).squeeze_(1)
        s_rel = self.rel_atten(rel_d, rel_h).permute(0, 2, 3, 1)

        loss = None
        if labels is not None:
            loss = self.loss_func(self, s_arc, s_rel, head, labels, mask)

        decode_s_arc = s_arc
        if mask is not None:
            activate_word_mask = torch.cat([mask[:, :1], mask], dim=1)
            activate_word_mask = activate_word_mask.unsqueeze(-1).expand_as(s_arc)
            activate_word_mask = activate_word_mask & activate_word_mask.transpose(-1, -2)
            decode_s_arc = s_arc.masked_fill(~activate_word_mask, float('-inf'))

        return GraphResult(loss=loss, arc_logits=decode_s_arc, rel_logits=s_rel, src_arc_logits=s_arc)


class TransformerBiaffine(BaseModule):
    def __init__(self, hparams, loss_func=dep_loss, config=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.transformer = AutoModel.from_pretrained(self.hparams.transformer)
        else:
            self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(self.hparams.dropout)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = BiaffineClassifier(
            hidden_size,
            label_num=self.hparams.num_labels,
            dropout=self.hparams.dropout,
            arc_hidden_size=self.hparams.arc_hidden_size,
            rel_hidden_size=self.hparams.rel_hidden_size,
            loss_interpolation=self.hparams.loss_interpolation,
            loss_func=loss_func,
            char_based=self.hparams.char_based
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--arc_hidden_size', type=int, default=500)
        parser.add_argument('--rel_hidden_size', type=int, default=200)
        parser.add_argument('--loss_interpolation', type=float, default=0.4)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_labels', type=int)
        parser.add_argument('--char_based', action='store_true')
        return parser

    def forward(
            self,
            input_ids=None,
            logits_mask=None,
            attention_mask=None,
            word_index=None,
            word_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            head=None,
            labels=None
    ) -> GraphResult:
        hidden_states = self.transformer(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        sequence_output = hidden_states[0]
        sequence_output = self.dropout(sequence_output)

        return self.classifier(
            input=sequence_output,
            attention_mask=attention_mask,
            word_index=word_index,
            word_attention_mask=word_attention_mask,
            head=head,
            labels=labels
        )
