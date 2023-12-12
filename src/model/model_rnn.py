from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn as nn
import torch
from torch.nn import functional as F
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
from torch.nn import TransformerEncoderLayer
from src.model.attention import MultiheadAttention
from random import random, randint, randrange
from torch.nn import MSELoss
import math

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1)

        self.div_term = math.sqrt(hidden_size)

    def forward(self, last_hidden, decoder_inputs, encoder_outputs, attention_mask, is_training=True):
        '''
        :param last_hidden: (1, batch, hidden)
        :param decoder_inputs: (batch, 1, hidden)
        :param encoder_outputs: (batch, seq_len, hidden)
        :return:
        '''
        batch_size = decoder_inputs.size(0)
        indexes = [e for e in range(batch_size)]
        key_encoder_outputs = self.dense1(encoder_outputs)
        value_encoder_outputs = self.dense2(encoder_outputs)
        # key : (batch, seq, hidden)
        # value : (batch, seq, hidden)

        output, hidden = self.decoder(decoder_inputs, hx=last_hidden)
        # output : (batch, 1, hidden)
        # hidden : (1, batch, hidden)

        t_encoder_outputs = key_encoder_outputs.transpose(1, 2)
        # t_encoder_outputs : (batch, hidden, seq)

        attn_outputs = output.bmm(t_encoder_outputs) + attention_mask
        # attn_outputs : (batch, 1, seq_len)

        # attn_alignment = F.softmax(attn_outputs, -1)
        if is_training:
            # attn_alignment = F.gumbel_softmax(attn_outputs, tau=1, hard=False, dim=-1)
            attn_alignment = F.softmax(attn_outputs, -1)
        else:
            attn_alignment = F.softmax(attn_outputs, -1)
        # attn_alignment : (batch, 1, seq_len)

        evidence_sentence = attn_alignment.argmax(-1).squeeze(1)
        #if is_training:
        attention_mask[indexes, 0, evidence_sentence] = -1e10
        context = attn_alignment.bmm(value_encoder_outputs)
        # context : (batch, 1, hidden)

        hidden_states = torch.cat([context, output], -1)

        result = self.dense3(hidden_states)
        return result, hidden, evidence_sentence, attn_outputs, attention_mask

class SampledAttentionDecoder1204(nn.Module):
    def __init__(self, hidden_size):
        super(SampledAttentionDecoder1204, self).__init__()
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1)

        self.div_term = math.sqrt(hidden_size)

    def forward(self, last_hidden, decoder_inputs, encoder_outputs, attention_mask, is_training=True, is_sample=False):
        '''
        :param last_hidden: (1, batch, hidden)
        :param decoder_inputs: (batch, 1, hidden)
        :param encoder_outputs: (batch, seq_len, hidden)
        :return:
        '''
        batch_size = decoder_inputs.size(0)
        indexes = [e for e in range(batch_size)]
        key_encoder_outputs = self.dense1(encoder_outputs)
        value_encoder_outputs = self.dense2(encoder_outputs)
        # key : (batch, seq, hidden)
        # value : (batch, seq, hidden)

        output, hidden = self.decoder(decoder_inputs, hx=last_hidden)
        # output : (batch, 1, hidden)
        # hidden : (1, batch, hidden)

        t_encoder_outputs = key_encoder_outputs.transpose(1, 2)
        # t_encoder_outputs : (batch, hidden, seq)

        attn_outputs = output.bmm(t_encoder_outputs) + attention_mask
        # attn_outputs : (batch, 1, seq_len)

        # attn_alignment = F.softmax(attn_outputs, -1)

        if is_sample:
            attn_alignment = F.gumbel_softmax(attn_outputs, tau=1, hard=False, dim=-1)
        # a = torch.sum(attn_alignment1)
        else:
            attn_alignment = F.softmax(attn_outputs, -1)
            # b = torch.sum(attn_alignment)

        # attn_alignment : (batch, 1, seq_len)

        evidence_sentence = attn_alignment.argmax(-1).squeeze(1)
        #if is_training:
        attention_mask[indexes, 0, evidence_sentence] = -1e10
        context = attn_alignment.bmm(value_encoder_outputs)
        # context : (batch, 1, hidden)

        hidden_states = torch.cat([context, output], -1)

        result = self.dense3(hidden_states)
        return result, hidden, evidence_sentence, attn_outputs, attention_mask
class SampledAttentionDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(SampledAttentionDecoder, self).__init__()
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1)

        self.div_term = math.sqrt(hidden_size)
    def forward(self, last_hidden, decoder_inputs, encoder_outputs, attention_mask):
        '''
        :param last_hidden: (1, batch, hidden)
        :param decoder_inputs: (batch, 1, hidden)
        :param encoder_outputs: (batch, seq_len, hidden)
        :return:
        '''
        batch_size = decoder_inputs.size(0)

        key_encoder_outputs = self.dense1(encoder_outputs)
        value_encoder_outputs = self.dense2(encoder_outputs)
        # key : (batch, seq, hidden)
        # value : (batch, seq, hidden)

        output, hidden = self.decoder(decoder_inputs, hx=last_hidden)
        # output : (batch, 1, hidden)
        # hidden : (1, batch, hidden)

        t_encoder_outputs = key_encoder_outputs.transpose(1, 2)
        # t_encoder_outputs : (batch, hidden, seq)

        attn_outputs = output.bmm(t_encoder_outputs)  + attention_mask
        # attn_outputs : (batch, 1, seq_len)

        attn_alignment = F.gumbel_softmax(attn_outputs, tau=1, hard=False, dim=-1)
        # attn_alignment = F.softmax(attn_outputs, dim=-1)
        # attn_alignment : (batch, 1, seq_len)

        evidence_sentence = torch.argmax(attn_alignment, -1).squeeze(1)


        # attention_mask[indexes, 0, evidence_sentence] = -1e10

        for idx in range(len(attention_mask)):
            attention_mask[idx, 0, evidence_sentence[idx]] = -1e10
        aa = attention_mask.tolist()
        context = attn_alignment.bmm(value_encoder_outputs)
        # context : (batch, 1, hidden)

        hidden_states = torch.cat([context, output], -1)

        result = self.dense3(hidden_states)
        return result, hidden, evidence_sentence, attn_outputs.squeeze(1), attention_mask


class ElectraForQuestionAnswering_sent_evidence_trm_sampling_1016(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_sent_evidence_trm_sampling_1016, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num
        self.num_samples = config.num_samples
        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = AttentionDecoder(self.hidden_size)

        # ELECTRA weight 초기화
        self.init_weights()

    def extract_answerable_sent_idx(self, answer_positions, sent_masks):
        batch_size = sent_masks.size(0)
        answerable_sent_id = torch.gather(sent_masks, 1, answer_positions.unsqueeze(-1)).squeeze(-1)
        expanded_answerable_sent_id = F.one_hot(answerable_sent_id, num_classes=self.max_sent_num).reshape(
            batch_size * self.max_sent_num)
        expanded_answerable_sent_id[0::self.max_sent_num] = 0
        answerable_sent_ids= torch.where(expanded_answerable_sent_id ==1)[0]
        return answerable_sent_ids

    def _path_generate(self, hop_score, p_mask, n_mask, answerable_sent_num=None):
        if answerable_sent_num:
            n_negative_hop_score = hop_score + n_mask
            n_positive_hop_score = hop_score + p_mask
            _, n_hop_negative_path_idx = torch.sort(n_negative_hop_score, descending=True)
            n_hop_negative_path_idx = n_hop_negative_path_idx[:3]
            _, n_hop_positive_path_idx = torch.sort(n_positive_hop_score, descending=True)
            n_hop_positive_path_idx = n_hop_positive_path_idx[:answerable_sent_num]
            path_idx = torch.cat([n_hop_negative_path_idx, n_hop_positive_path_idx])
            path_label = torch.cat([torch.ones([3]).float(), torch.ones([answerable_sent_num])]).cuda()
            path_logits = torch.gather(hop_score, 0, path_idx)
            return path_logits, path_idx, path_label


        else:
            n_hop_score = hop_score + n_mask
            _, n_hop_path_idx = torch.sort(n_hop_score, descending=True)
            n_hop_path_idx = n_hop_path_idx[:1]
            path_logits = torch.gather(hop_score, 0, n_hop_path_idx)
            return path_logits, n_hop_path_idx, None

    def _cross_entropy(self, logits):
        loss1 = -(F.log_softmax(logits, dim=-1))
        # print(to_list(loss1))
        loss = loss1
        return loss

    def forward(
            self,
            input_ids=None,

            ##################
            attention_mask=None,
            token_type_ids=None,
            sent_masks=None,
            ##############

            start_positions=None,
            end_positions=None,
    ):

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        batch_size = input_ids.size(0)

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        is_training = False
        if start_positions is not None:
            is_training=True

        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_length, hidden_size]

        cls_output = sequence_output[:, 0, :]
        # cls_output : [batch, hidden]

        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        sentence_masks[:, 0, :] = sentence_masks[:, 0, :] * attention_mask
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]

        div_term = div_term.masked_fill(div_term == 0, 1e-10)

        attention_masks = div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).view(1,
                                                                                                            batch_size * self.max_sent_num).bool()
        sentence_representation = sentence_masks.bmm(sequence_output)
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]

        sentence_representation = self.dropout(
            sentence_representation.reshape(1, batch_size * self.max_sent_num, self.hidden_size))

        attention_mask = attention_masks.float()
        # [0, 0, 0, 1, 1,      [1, 0, 0, 0, 0
        #  0, 0, 0, 0, 1]       1, 0, 0, 0, 0]


        last_hidden = None
        decoder_inputs = torch.sum(sentence_representation[:, 0::self.max_sent_num, :], 1, keepdim=True) / batch_size
        encoder_outputs = sentence_representation
        attention_mask[:, 0::self.max_sent_num] = 1
        attention_mask = attention_mask.masked_fill(attention_mask == 1, -1e10).masked_fill(attention_mask == 0, 0)
        attention_mask = attention_mask.unsqueeze(0)
        evidence_sentences = []
        for evidence_step in range(3):
            decoder_inputs, last_hidden, evidence_sentence,attn_outputs, attention_mask = self.gru(last_hidden, decoder_inputs, encoder_outputs, attention_mask, is_training)
            evidence_sentences.append(evidence_sentence)
        evidence_vector = decoder_inputs.squeeze().unsqueeze(-1)
        evidence_sentences = torch.stack(evidence_sentences, 0)

        evidence_sentences = evidence_sentences.unsqueeze(0).expand(batch_size, -1, -1)
        start_representation = self.start_dense(sequence_output)
        end_representation = self.end_dense(sequence_output)

        start_logits = start_representation.matmul(evidence_vector)
        end_logits = end_representation.matmul(evidence_vector)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # outputs = (start_logits, end_logits)
        outputs = (start_logits, end_logits, evidence_sentences.squeeze(-1)) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언
            loss_fct = CrossEntropyLoss(reduction='none')

            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 최종 loss 계산
            span_loss = (start_loss + end_loss) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (span_loss, ) + outputs

        return outputs  # (loss), start_logits, end_logits

class ElectraForQuestionAnswering_sent_evidence_trm_sampling_1028(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_sent_evidence_trm_sampling_1028, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num
        self.num_samples = config.num_samples
        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = AttentionDecoder(self.hidden_size)

        # ELECTRA weight 초기화
        self.init_weights()

    def extract_answerable_sent_idx(self, answer_positions, sent_masks):
        batch_size = sent_masks.size(0)
        answerable_sent_id = torch.gather(sent_masks, 1, answer_positions.unsqueeze(-1)).squeeze(-1)
        expanded_answerable_sent_id = F.one_hot(answerable_sent_id, num_classes=self.max_sent_num).reshape(
            batch_size * self.max_sent_num)
        expanded_answerable_sent_id[0::self.max_sent_num] = 0
        answerable_sent_ids= torch.where(expanded_answerable_sent_id ==1)[0]
        return answerable_sent_ids

    def _path_generate(self, hop_score, p_mask, n_mask, answerable_sent_num=None):
        if answerable_sent_num:
            n_negative_hop_score = hop_score + n_mask
            n_positive_hop_score = hop_score + p_mask
            _, n_hop_negative_path_idx = torch.sort(n_negative_hop_score, descending=True)
            n_hop_negative_path_idx = n_hop_negative_path_idx[:3]
            _, n_hop_positive_path_idx = torch.sort(n_positive_hop_score, descending=True)
            n_hop_positive_path_idx = n_hop_positive_path_idx[:answerable_sent_num]
            path_idx = torch.cat([n_hop_negative_path_idx, n_hop_positive_path_idx])
            path_label = torch.cat([torch.ones([3]).float(), torch.ones([answerable_sent_num])]).cuda()
            path_logits = torch.gather(hop_score, 0, path_idx)
            return path_logits, path_idx, path_label


        else:
            n_hop_score = hop_score + n_mask
            _, n_hop_path_idx = torch.sort(n_hop_score, descending=True)
            n_hop_path_idx = n_hop_path_idx[:1]
            path_logits = torch.gather(hop_score, 0, n_hop_path_idx)
            return path_logits, n_hop_path_idx, None

    def _cross_entropy(self, logits):
        loss1 = -(F.log_softmax(logits, dim=-1))
        # print(to_list(loss1))
        loss = loss1
        return loss

    def forward(
            self,
            input_ids=None,

            ##################
            attention_mask=None,
            token_type_ids=None,
            sent_masks=None,
            ##############

            start_positions=None,
            end_positions=None,
    ):

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        is_training = False
        if start_positions is not None:
            is_training=True

        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_length, hidden_size]

        cls_output = sequence_output[:, 0, :]
        # cls_output : [batch, hidden]
        all_cls_output = torch.sum(cls_output, dim=0, keepdim=True) / batch_size
        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        sentence_masks[:, 0, :] = sentence_masks[:, 0, :] * attention_mask
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]

        div_term = div_term.masked_fill(div_term == 0, 1e-10)

        attention_masks = div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).view(1,
                                                                                                            batch_size * self.max_sent_num).bool()
        sentence_representation = sentence_masks.bmm(sequence_output)
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]

        sentence_representation = self.dropout(
            sentence_representation.reshape(1, batch_size * self.max_sent_num, self.hidden_size))

        attention_masks = attention_masks.float()
        # [0, 0, 0, 1, 1,      [1, 0, 0, 0, 0
        #  0, 0, 0, 0, 1]       1, 0, 0, 0, 0]


        last_hidden = None
        # decoder_inputs = all_cls_output.unsqueeze(1)
        decoder_inputs = torch.sum(sentence_representation[:, 0::self.max_sent_num, :], 1, keepdim=True) / batch_size
        encoder_outputs = sentence_representation
        attention_masks[:, 0::self.max_sent_num] = 1
        mm = 1-attention_masks
        mm = mm.unsqueeze(1).expand(-1, 3, -1)
        attention_masks = attention_masks.masked_fill(attention_masks == 1, -1e10).masked_fill(attention_masks == 0, 0).unsqueeze(0)
        # if is_training:
        #     decoder_inputs = decoder_inputs.expand(self.num_samples, -1, -1)
        #     encoder_outputs = encoder_outputs.expand(self.num_samples, -1, -1)
        #     attention_masks = attention_masks.expand(self.num_samples, -1, -1)
        #     attention_masks = attention_masks.clone().detach()
        evidence_sentences = []
        attention_scores = []
        for evidence_step in range(3):
            decoder_inputs, last_hidden, evidence_sentence, attention_score, attention_masks = self.gru(last_hidden, decoder_inputs, encoder_outputs, attention_masks)
            evidence_sentences.append(evidence_sentence)
            attention_scores.append(attention_score)
        evidence_vector = decoder_inputs.squeeze(1).transpose(0, 1)
        evidence_sentences = torch.stack(evidence_sentences, 0)
        attention_scores = torch.stack(attention_scores, 0)


        evidence_sentences = evidence_sentences.transpose(0, 1)
        attention_scores = attention_scores.transpose(0, 1)
        if is_training:
            evidence = evidence_sentences.unsqueeze(0)
        else:
            evidence = evidence_sentences.unsqueeze(0).expand(batch_size, -1, -1)

        start_representation = self.start_dense(sequence_output)
        end_representation = self.end_dense(sequence_output)

        start_logits = start_representation.matmul(evidence_vector)
        end_logits = end_representation.matmul(evidence_vector)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        # outputs = (start_logits, end_logits)
        outputs = (start_logits, end_logits, evidence.squeeze(1)) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언

            loss_fct = CrossEntropyLoss()

            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 최종 loss 계산
            span_loss = (start_loss + end_loss) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (span_loss, attention_scores, mm) + outputs

        return outputs  # (loss), start_logits, end_logits

class ElectraForQuestionAnswering_sent_evidence_final(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_sent_evidence_final, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num
        self.num_samples = config.num_samples
        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels+1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = AttentionDecoder(self.hidden_size)

        # ELECTRA weight 초기화
        self.init_weights()

    def extract_answerable_sent_idx(self, answer_positions, sent_masks):
        batch_size = sent_masks.size(0)
        answerable_sent_id = torch.gather(sent_masks, 1, answer_positions.unsqueeze(-1)).squeeze(-1)
        expanded_answerable_sent_id = F.one_hot(answerable_sent_id, num_classes=self.max_sent_num).reshape(
            batch_size * self.max_sent_num)
        expanded_answerable_sent_id[0::self.max_sent_num] = 0
        answerable_sent_ids= torch.where(expanded_answerable_sent_id ==1)[0]
        return answerable_sent_ids

    def _path_generate(self, hop_score, p_mask, n_mask, answerable_sent_num=None):
        if answerable_sent_num:
            n_negative_hop_score = hop_score + n_mask
            n_positive_hop_score = hop_score + p_mask
            _, n_hop_negative_path_idx = torch.sort(n_negative_hop_score, descending=True)
            n_hop_negative_path_idx = n_hop_negative_path_idx[:3]
            _, n_hop_positive_path_idx = torch.sort(n_positive_hop_score, descending=True)
            n_hop_positive_path_idx = n_hop_positive_path_idx[:answerable_sent_num]
            path_idx = torch.cat([n_hop_negative_path_idx, n_hop_positive_path_idx])
            path_label = torch.cat([torch.ones([3]).float(), torch.ones([answerable_sent_num])]).cuda()
            path_logits = torch.gather(hop_score, 0, path_idx)
            return path_logits, path_idx, path_label


        else:
            n_hop_score = hop_score + n_mask
            _, n_hop_path_idx = torch.sort(n_hop_score, descending=True)
            n_hop_path_idx = n_hop_path_idx[:1]
            path_logits = torch.gather(hop_score, 0, n_hop_path_idx)
            return path_logits, n_hop_path_idx, None

    def _cross_entropy(self, logits):
        loss1 = -(F.log_softmax(logits, dim=-1))
        # print(to_list(loss1))
        loss = loss1
        return loss

    def forward(
            self,
            input_ids=None,

            ##################
            attention_mask=None,
            token_type_ids=None,
            sent_masks=None,
            ##############
            question_type=None,
            start_positions=None,
            end_positions=None,
    ):

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        is_training = False
        if start_positions is not None:
            is_training=True

        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_length, hidden_size]

        cls_output = sequence_output[:, 0, :]
        # cls_output : [batch, hidden]
        all_cls_output = torch.sum(cls_output, dim=0, keepdim=True) / batch_size
        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        sentence_masks[:, 0, :] = sentence_masks[:, 0, :] * attention_mask
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]

        div_term = div_term.masked_fill(div_term == 0, 1e-10)

        attention_masks = div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).view(1,
                                                                                                            batch_size * self.max_sent_num).bool()
        sentence_representation = sentence_masks.bmm(sequence_output)
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]

        sentence_representation = self.dropout(
            sentence_representation.reshape(1, batch_size * self.max_sent_num, self.hidden_size))

        attention_masks = attention_masks.float()
        # [0, 0, 0, 1, 1,      [1, 0, 0, 0, 0
        #  0, 0, 0, 0, 1]       1, 0, 0, 0, 0]


        last_hidden = None
        # decoder_inputs = all_cls_output.unsqueeze(1)
        decoder_inputs = torch.sum(sentence_representation[:, 0::self.max_sent_num, :], 1, keepdim=True) / batch_size
        encoder_outputs = sentence_representation
        attention_masks[:, 0::self.max_sent_num] = 1
        mm = 1-attention_masks
        mm = mm.unsqueeze(1).expand(-1, 3, -1)
        attention_masks = attention_masks.masked_fill(attention_masks == 1, -1e10).masked_fill(attention_masks == 0, 0).unsqueeze(0)
        # if is_training:
        #     decoder_inputs = decoder_inputs.expand(self.num_samples, -1, -1)
        #     encoder_outputs = encoder_outputs.expand(self.num_samples, -1, -1)
        #     attention_masks = attention_masks.expand(self.num_samples, -1, -1)
        #     attention_masks = attention_masks.clone().detach()
        evidence_sentences = []
        attention_scores = []
        for evidence_step in range(3):
            decoder_inputs, last_hidden, evidence_sentence, attention_score, attention_masks = self.gru(last_hidden, decoder_inputs, encoder_outputs, attention_masks)
            evidence_sentences.append(evidence_sentence)
            attention_scores.append(attention_score)
        evidence_vector = decoder_inputs.squeeze(1).transpose(0, 1)
        evidence_sentences = torch.stack(evidence_sentences, 0)
        attention_scores = torch.stack(attention_scores, 0)


        evidence_sentences = evidence_sentences.transpose(0, 1)
        attention_scores = attention_scores.transpose(0, 1)
        if is_training:
            evidence = evidence_sentences.unsqueeze(0)
        else:
            evidence = evidence_sentences.unsqueeze(0).expand(batch_size, -1, -1)

        start_representation = self.start_dense(sequence_output)
        end_representation = self.end_dense(sequence_output)

        start_logits = start_representation.matmul(evidence_vector)
        end_logits = end_representation.matmul(evidence_vector)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        qt_logits = self.qa_outputs(decoder_inputs.squeeze(1))
        # outputs = (start_logits, end_logits)
        if not is_training:
            qt_logits = torch.argmax(qt_logits.expand(batch_size, -1), -1)
        outputs = (start_logits, end_logits, qt_logits, evidence.squeeze(1)) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언

            loss_fct = CrossEntropyLoss()

            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qt_loss = loss_fct(qt_logits, question_type)
            # 최종 loss 계산
            span_loss = (start_loss + end_loss) / 2 + qt_loss

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (span_loss, attention_scores, mm) + outputs

        return outputs  # (loss), start_logits, end_logits

class ElectraForQuestionAnswering_1204(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_1204, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num
        self.max_dec_len = config.max_dec_len
        self.num_samples = config.num_samples
        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = SampledAttentionDecoder1204(self.hidden_size)

        # ELECTRA weight 초기화
        self.init_weights()

    def extract_answerable_sent_idx(self, answer_positions, sent_masks):
        batch_size = sent_masks.size(0)
        answerable_sent_id = torch.gather(sent_masks, 1, answer_positions.unsqueeze(-1)).squeeze(-1)
        expanded_answerable_sent_id = F.one_hot(answerable_sent_id, num_classes=self.max_sent_num).reshape(
            batch_size * self.max_sent_num)
        expanded_answerable_sent_id[0::self.max_sent_num] = 0
        answerable_sent_ids= torch.where(expanded_answerable_sent_id ==1)[0]
        return answerable_sent_ids

    def _path_generate(self, hop_score, p_mask, n_mask, answerable_sent_num=None):
        if answerable_sent_num:
            n_negative_hop_score = hop_score + n_mask
            n_positive_hop_score = hop_score + p_mask
            _, n_hop_negative_path_idx = torch.sort(n_negative_hop_score, descending=True)
            n_hop_negative_path_idx = n_hop_negative_path_idx[:3]
            _, n_hop_positive_path_idx = torch.sort(n_positive_hop_score, descending=True)
            n_hop_positive_path_idx = n_hop_positive_path_idx[:answerable_sent_num]
            path_idx = torch.cat([n_hop_negative_path_idx, n_hop_positive_path_idx])
            path_label = torch.cat([torch.ones([3]).float(), torch.ones([answerable_sent_num])]).cuda()
            path_logits = torch.gather(hop_score, 0, path_idx)
            return path_logits, path_idx, path_label


        else:
            n_hop_score = hop_score + n_mask
            _, n_hop_path_idx = torch.sort(n_hop_score, descending=True)
            n_hop_path_idx = n_hop_path_idx[:1]
            path_logits = torch.gather(hop_score, 0, n_hop_path_idx)
            return path_logits, n_hop_path_idx, None

    def _cross_entropy(self, logits):
        loss1 = -(F.log_softmax(logits, dim=-1))
        # print(to_list(loss1))
        loss = loss1
        return loss

    def forward(
            self,
            input_ids=None,

            ##################
            attention_mask=None,
            token_type_ids=None,
            sent_masks=None,
            ##############

            start_positions=None,
            end_positions=None,
    ):

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        is_training = False
        if start_positions is not None:
            is_training=True

        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_length, hidden_size]

        cls_output = sequence_output[:, 0, :]
        # cls_output : [batch, hidden]
        all_cls_output = torch.sum(cls_output, dim=0, keepdim=True) / batch_size
        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        sentence_masks[:, 0, :] = sentence_masks[:, 0, :] * attention_mask
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]

        div_term = div_term.masked_fill(div_term == 0, 1e-10)

        attention_masks = div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).view(1,
                                                                                                            batch_size * self.max_sent_num).bool()
        sentence_representation = sentence_masks.bmm(sequence_output)
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]

        sentence_representation = self.dropout(
            sentence_representation.reshape(1, batch_size * self.max_sent_num, self.hidden_size))

        attention_masks = attention_masks.float()
        # [0, 0, 0, 1, 1,      [1, 0, 0, 0, 0
        #  0, 0, 0, 0, 1]       1, 0, 0, 0, 0]


        last_hidden = None
        # decoder_inputs = all_cls_output.unsqueeze(1)
        decoder_inputs = torch.sum(sentence_representation[:, 0::self.max_sent_num, :], 1, keepdim=True) / batch_size
        encoder_outputs = sentence_representation
        attention_masks[:, 0::self.max_sent_num] = 1
        mm = 1-attention_masks
        mm = mm.unsqueeze(1).expand(-1, self.max_dec_len, -1)
        attention_masks = attention_masks.masked_fill(attention_masks == 1, -1e10).masked_fill(attention_masks == 0, 0).unsqueeze(0)
        if is_training:
            decoder_inputs = decoder_inputs.expand(self.num_samples, -1, -1)
            encoder_outputs = encoder_outputs.expand(self.num_samples, -1, -1)
            attention_masks = attention_masks.expand(self.num_samples, -1, -1)
            attention_masks = attention_masks.clone().detach()
        evidence_sentences = []
        attention_scores = []
        for evidence_step in range(self.max_dec_len):
            decoder_inputs, last_hidden, evidence_sentence, attention_score, attention_masks = self.gru(last_hidden, decoder_inputs, encoder_outputs, attention_masks)
            evidence_sentences.append(evidence_sentence)
            attention_scores.append(attention_score)
        evidence_vector = decoder_inputs.squeeze(1).transpose(0, 1)
        evidence_sentences = torch.stack(evidence_sentences, 0)
        attention_scores = torch.stack(attention_scores, 0)


        evidence_sentences = evidence_sentences.transpose(0, 1)
        attention_scores = attention_scores.transpose(0, 1)
        if not is_training:
            evidence = evidence_sentences.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            evidence = evidence_sentences
        start_representation = self.start_dense(sequence_output)
        end_representation = self.end_dense(sequence_output)

        start_logits = start_representation.matmul(evidence_vector)
        end_logits = end_representation.matmul(evidence_vector)
        if not is_training:
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)


        # outputs = (start_logits, end_logits)
        outputs = (start_logits, end_logits, evidence.squeeze(1)) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언

            loss_fct = CrossEntropyLoss()
            start_logits = start_logits.permute(2, 0, 1)
            end_logits = end_logits.permute(2, 0, 1)
            start_positions = start_positions.unsqueeze(0).expand(self.num_samples, -1)
            end_positions = end_positions.unsqueeze(0).expand(self.num_samples, -1)
            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits.reshape(batch_size*self.num_samples, sequence_length), start_positions.reshape(batch_size*self.num_samples))
            end_loss = loss_fct(end_logits.reshape(batch_size*self.num_samples, sequence_length), end_positions.reshape(batch_size*self.num_samples))

            # 최종 loss 계산
            span_loss = (start_loss + end_loss) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (span_loss, attention_scores, mm) + outputs

        return outputs  # (loss), start_logits, end_logits
class ElectraForQuestionAnswering_1208(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_1208, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num
        self.max_dec_len = config.max_dec_len
        self.num_samples = config.num_samples
        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = SampledAttentionDecoder1204(self.hidden_size)

        # ELECTRA weight 초기화
        self.init_weights()

    def forward(
            self,
            input_ids=None,

            ##################
            attention_mask=None,
            token_type_ids=None,
            sent_masks=None,
            ##############

            start_positions=None,
            end_positions=None,
    ):

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        is_training = False
        if start_positions is not None:
            is_training=True

        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_length, hidden_size]

        cls_output = sequence_output[:, 0, :]
        # cls_output : [batch, hidden]
        all_cls_output = torch.sum(cls_output, dim=0, keepdim=True) / batch_size
        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        sentence_masks[:, 0, :] = sentence_masks[:, 0, :] * attention_mask
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]

        div_term = div_term.masked_fill(div_term == 0, 1e-10)

        attention_masks = div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).view(1,
                                                                                                            batch_size * self.max_sent_num).bool()
        sentence_representation = sentence_masks.bmm(sequence_output)
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]

        sentence_representation = self.dropout(
            sentence_representation.reshape(1, batch_size * self.max_sent_num, self.hidden_size))

        attention_masks = attention_masks.float()
        # [0, 0, 0, 1, 1,      [1, 0, 0, 0, 0
        #  0, 0, 0, 0, 1]       1, 0, 0, 0, 0]


        last_hidden = None
        sampled_last_hidden = None
        # decoder_inputs = all_cls_output.unsqueeze(1)
        decoder_inputs = torch.sum(sentence_representation[:, 0::self.max_sent_num, :], 1, keepdim=True) / batch_size
        sampled_decoder_inputs = torch.sum(sentence_representation[:, 0::self.max_sent_num, :], 1, keepdim=True) / batch_size
        encoder_outputs = sentence_representation
        attention_masks[:, 0::self.max_sent_num] = 1
        mm = 1-attention_masks
        mm = mm.unsqueeze(1).expand(-1, self.max_dec_len, -1)
        attention_masks = attention_masks.masked_fill(attention_masks == 1, -1e10).masked_fill(attention_masks == 0, 0).unsqueeze(0)
        sampled_attention_masks = attention_masks.clone().detach()
        evidence_sentences = []
        attention_scores = []
        sampled_evidence_sentences = []
        sampled_attention_scores = []
        for evidence_step in range(self.max_dec_len):
            decoder_inputs, last_hidden, evidence_sentence, attention_score, attention_masks = self.gru(last_hidden,
                                                                                                        decoder_inputs,
                                                                                                        encoder_outputs,
                                                                                                        attention_masks)
            evidence_sentences.append(evidence_sentence)
            attention_scores.append(attention_score)
        evidence_vector = decoder_inputs.squeeze(1).transpose(0, 1)
        evidence_sentences = torch.stack(evidence_sentences, 0)
        attention_scores = torch.stack(attention_scores, 0)
        # if is_training:
        #     sampled_decoder_inputs = sampled_decoder_inputs.expand(self.num_samples, -1, -1)
        #     encoder_outputs = encoder_outputs.expand(self.num_samples, -1, -1)
        #     sampled_attention_masks = sampled_attention_masks.expand(self.num_samples, -1, -1)
        #     sampled_attention_masks = sampled_attention_masks.clone().detach()
        #
        #     for evidence_step in range(self.max_dec_len):
        #         sampled_decoder_inputs, sampled_last_hidden, evidence_sentence, attention_score, sampled_attention_masks = self.gru(
        #             sampled_last_hidden, sampled_decoder_inputs, encoder_outputs, sampled_attention_masks, is_sample=True)
        #         sampled_evidence_sentences.append(evidence_sentence)
        #         sampled_attention_scores.append(attention_score)
        #     sampled_evidence_vector = sampled_decoder_inputs.squeeze(1).transpose(0, 1)
        #     sampled_evidence_sentences = torch.stack(sampled_evidence_sentences, 0)
        #     sampled_attention_scores = torch.stack(sampled_attention_scores, 0)
        #
        #     evidence_vector = torch.cat([evidence_vector, sampled_evidence_vector], -1)
        #     evidence_sentences = torch.cat([evidence_sentences, sampled_evidence_sentences], -1)
        #     attention_scores = torch.cat([attention_scores, sampled_attention_scores], 1)

        evidence_sentences = evidence_sentences.transpose(0, 1)
        attention_scores = attention_scores.transpose(0, 1)
        if not is_training:
            evidence = evidence_sentences.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            evidence = evidence_sentences
        start_representation = self.start_dense(sequence_output)
        end_representation = self.end_dense(sequence_output)

        start_logits = start_representation.matmul(evidence_vector)
        end_logits = end_representation.matmul(evidence_vector)
        if not is_training:
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)


        # outputs = (start_logits, end_logits)
        outputs = (start_logits, end_logits, evidence.squeeze(1)) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언

            loss_fct = CrossEntropyLoss()
            start_logits = start_logits.permute(2, 0, 1)
            end_logits = end_logits.permute(2, 0, 1)
            start_positions = start_positions.unsqueeze(0).expand((1+self.num_samples), -1)
            end_positions = end_positions.unsqueeze(0).expand((1+self.num_samples), -1)
            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits.reshape(batch_size*(1+self.num_samples), sequence_length), start_positions.reshape(batch_size*(1+self.num_samples)))
            end_loss = loss_fct(end_logits.reshape(batch_size*(1+self.num_samples), sequence_length), end_positions.reshape(batch_size*(1+self.num_samples)))

            # 최종 loss 계산
            span_loss = (start_loss + end_loss) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (span_loss, attention_scores, mm) + outputs

        return outputs  # (loss), start_logits, end_logits
