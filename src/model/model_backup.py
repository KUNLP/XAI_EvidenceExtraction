from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from torch.nn import functional as F
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
import numpy as np
import math
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1)
    def forward(self, last_hidden, decoder_inputs, encoder_outputs, attention_mask):
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

        attn_alignment = F.softmax(attn_outputs, -1)
        # attn_alignment : (batch, 1, seq_len)

        evidence_sentence = attn_alignment.argmax(-1).squeeze(1)
        attention_mask[indexes, :, evidence_sentence] = -1e10
        context = attn_alignment.bmm(value_encoder_outputs)
        # context : (batch, 1, hidden)

        hidden_states = torch.cat([context, output], -1)

        result = self.dense3(hidden_states)
        return result, hidden, evidence_sentence, attention_mask
class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.div_dim = np.sqrt(hidden_dim)

        self.q_dense = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.k_dense = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.v_dense = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_training=False):
        bzs, seq_len, hidden_dim = query.size()

        attn_mask = attn_mask.unsqueeze(0).expand(bzs, -1, -1)


        key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
        key_padding_mask = key_padding_mask + key_padding_mask.transpose(1, 2)

        dense_q = self.q_dense(query)
        dense_k = self.k_dense(key)
        dense_v = self.v_dense(value)

        q = self.dropout(dense_q)
        # (batch, seq_len, hidden)
        k = self.dropout(dense_k).transpose(1, 2)
        # (batch,  hidden, seq_len)
        v = self.dropout(dense_v)
        # (batch, seq_len, hidden)

        attn_alignment = q.bmm(k)
        attn_alignment = attn_alignment.masked_fill(key_padding_mask==True, -1e10).masked_fill(attn_mask==True, -1e10)

        if is_training:
            attn_weights = F.gumbel_softmax(attn_alignment, tau=0.1, hard=True, dim=-1)
        else:
            aa = attn_alignment.tolist()
            attn_weights = F.softmax(attn_alignment, -1)
            a = attn_weights.tolist()
            attn_weights = torch.argmax(attn_weights, -1)
            attn_weights = F.one_hot(attn_weights, seq_len).float().cuda()


        attn_outputs = attn_weights.bmm(v) / self.div_dim
        # (batch, seq_len, seq_len)

        return attn_outputs, attn_weights




class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, src, key, value, src_mask = None, src_key_padding_mask = None, is_training=False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """


        src2, attn_weight = self.self_attn(src, key, value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, is_training=is_training)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weight

class ElectraForQuestionAnswering_sent_evidence_trm_sampling(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_sent_evidence_trm_sampling, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num


        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_2 = CustomTransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
        self.layer_3 = CustomTransformerEncoderLayer(d_model=config.hidden_size, nhead=8)

        self.hop1_dense = nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size)
        self.hop2_dense = nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size)
        self.hop3_dense = nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size)
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        # ELECTRA weight 초기화
        self.init_weights()

    def _make_mask(self, answer_positions, attention_masks, sent_masks):
        batch_size = sent_masks.size(0)
        answerable_sent_id = torch.gather(sent_masks, 1, answer_positions.unsqueeze(-1)).squeeze(-1)
        expanded_answerable_sent_id = F.one_hot(answerable_sent_id, num_classes=self.max_sent_num).reshape(
            batch_size * self.max_sent_num)
        expanded_answerable_sent_id[0::self.max_sent_num] = 0
        answerable_sent_mask = expanded_answerable_sent_id.masked_fill(expanded_answerable_sent_id == 1, -1e10)
        attention_masks = attention_masks.float()
        pad_mask = attention_masks.masked_fill(attention_masks == 1, -1e10)[0]
        pad_mask[0::self.max_sent_num] = -1e10
        n_mask = pad_mask + answerable_sent_mask
        p_mask = expanded_answerable_sent_id
        answerable_sent_num = torch.sum(p_mask)
        p_mask = p_mask.masked_fill(p_mask == 0, -1e10).masked_fill(p_mask==1, 0)

        return p_mask, n_mask, answerable_sent_num

    def _path_generate(self, hop_score, p_mask, n_mask, answerable_sent_num=None):
        if answerable_sent_num:
            n_negative_hop_score = hop_score + n_mask
            n_positive_hop_score = hop_score + p_mask
            _, n_hop_negative_path_idx = torch.sort(n_negative_hop_score, descending=True)
            n_hop_negative_path_idx = n_hop_negative_path_idx[:3]
            _, n_hop_positive_path_idx = torch.sort(n_positive_hop_score, descending=True)
            n_hop_positive_path_idx = n_hop_positive_path_idx[:answerable_sent_num]
            path_idx = torch.cat([n_hop_negative_path_idx, n_hop_positive_path_idx])
            path_label = torch.cat([torch.zeros([3]).float(), torch.ones([answerable_sent_num])]).cuda()
            path_logits = torch.gather(hop_score, 0, path_idx)
            return path_logits, path_idx, path_label


        else:
            n_hop_score = hop_score + n_mask
            _, n_hop_path_idx = torch.sort(n_hop_score, descending=True)
            n_hop_path_idx = n_hop_path_idx[:1]
            path_logits = torch.gather(hop_score, 0, n_hop_path_idx)
            return path_logits, n_hop_path_idx, None
    def _cross_entropy(self, logits, labels):
        loss1 = -(F.log_softmax(logits, dim=-1) * labels)
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
            train_mode=False
    ):
        # if not train_mode:
        #     self.electra.req
        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        batch_size = input_ids.size(0)
        sentence_size = input_ids.size(1)

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        is_training = True if start_positions is not None else False
        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_length, hidden_size]

        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]

        div_term = div_term.masked_fill(div_term==0, 1e-10)

        attention_masks = div_term.masked_fill(div_term!=1e-10, 0).masked_fill(div_term==1e-10, 1).view(1, batch_size*self.max_sent_num).bool()
        # attention_masks : [batch, sent_num, 1] -> [1, batch * sent_num]

        sentence_representation = sentence_masks.bmm(sequence_output)
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]

        hop_1_sentence_representation = self.dropout(sentence_representation.reshape(1, batch_size*self.max_sent_num, self.hidden_size))
        # sentence_representation : [batch*sent_num, 1, hidden]

        if start_positions is not None:
            p_mask, n_mask, answerable_sent_num = self._make_mask(start_positions, attention_masks, sent_masks)
        else:
            attention_mask = attention_masks.float()
            pad_mask = attention_mask.masked_fill(attention_mask == 1, -1e10)[0]
            pad_mask[0::self.max_sent_num] = -1e10
            p_mask = answerable_sent_num = None
            n_mask = pad_mask

        q_vectors = hop_1_sentence_representation[0, 0::self.max_sent_num, :]
        q_vectors = q_vectors.unsqueeze(1).expand(-1, self.max_sent_num, -1).reshape(1, batch_size*self.max_sent_num, -1)

        hop_1_representation = torch.cat([hop_1_sentence_representation, q_vectors], -1)
        dense_hop_1_representation = self.dropout(self.hop1_dense(hop_1_representation))

        mask = torch.zeros(self.max_sent_num*batch_size, self.max_sent_num*batch_size).float().cuda()

        mask[:, 0::self.max_sent_num] = 1
        hop_2_sentence_representation, attn_1_weight = self.layer_2(dense_hop_1_representation,
                                                                    hop_1_sentence_representation,
                                                                    hop_1_sentence_representation, src_mask=mask.bool(),
                                                                    src_key_padding_mask=attention_masks,
                                                                    is_training=is_training)
        hop_2_representation = torch.cat([hop_2_sentence_representation, q_vectors], -1)
        dense_hop_2_representation = self.dropout(self.hop2_dense(hop_2_representation))


        hop_3_sentence_representation, attn_2_weight = self.layer_3(dense_hop_2_representation,
                                                                    hop_1_sentence_representation,
                                                                    hop_1_sentence_representation, src_mask=mask.bool(),
                                                                    src_key_padding_mask=attention_masks,
                                                                    is_training=is_training)

        final_q_vector = q_vectors.transpose(1, 2)
        dense_hop_3_representation = self.dropout(self.dense(hop_3_sentence_representation))
        hop_3_score = dense_hop_3_representation.bmm(final_q_vector)[0, :, 0]
        hop_3_evidence_path_logits, hop_3_evidence_path_idx, hop_3_path_label = self._path_generate(hop_3_score, p_mask,
                                                                                                    n_mask,
                                                                                                    answerable_sent_num)
        all_hop_3_representation = self.dropout(self.hop3_dense(torch.cat([hop_3_sentence_representation, q_vectors], -1)))
        t_hop_3_sentence_representation = all_hop_3_representation.squeeze(0)
        hop_3_path_id = hop_3_evidence_path_idx.unsqueeze(-1).expand(-1, self.hidden_size)
        hop_3_evidence_representation = torch.gather(t_hop_3_sentence_representation, 0, hop_3_path_id)


        if hop_3_path_label is not None:
            evidence_path = hop_3_evidence_representation.transpose(1, 0)
            return_path=None
            # [15, hidden]
        else:
            evidence_path3 = torch.gather(t_hop_3_sentence_representation, 0, hop_3_path_id).transpose(0, 1)
            hop_3_sent = hop_3_evidence_path_idx[0]
            hop_2_sent = torch.argmax(attn_2_weight[0][hop_3_sent])
            hop_1_sent = torch.argmax(attn_1_weight[0][hop_2_sent])
            return_path = [hop_3_sent, hop_2_sent, hop_1_sent]
            return_path3 = torch.stack(return_path, 0)

            return_path = return_path3
            evidence_path = evidence_path3

        start_representation = self.dropout(self.start_dense(sequence_output))
        end_representation = self.dropout(self.end_dense(sequence_output))

        start_evidences = torch.matmul(start_representation, evidence_path)
        end_evidences = torch.matmul(end_representation, evidence_path)

        if hop_3_path_label is not None:
            start_logits =  start_evidences
            end_logits =  end_evidences
            start_positions = start_positions.unsqueeze(0).expand(start_evidences.size(-1), -1)
            end_positions = end_positions.unsqueeze(0).expand(start_evidences.size(-1), -1)
        else:
            start_evidences = start_evidences.squeeze(-1)
            end_evidences = end_evidences.squeeze(-1)
            start_logits =  start_evidences
            end_logits =  end_evidences
        if return_path is None:
            return_path = hop_3_evidence_path_idx.unsqueeze(0).expand(batch_size, -1)
        else:
            return_path= return_path.unsqueeze(0).expand(batch_size, -1)
        # outputs = (start_logits, end_logits)
        path_label = hop_3_path_label if start_positions is not None else None
        outputs = (start_logits, end_logits, return_path if start_positions is None else path_label) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언
            loss_fct = CrossEntropyLoss(reduction='none')

            start_logits = start_logits.permute(2, 0, 1)
            end_logits = end_logits.permute(2, 0, 1)

            hop_3_loss = self._cross_entropy(hop_3_evidence_path_logits, hop_3_path_label)
            path_label = path_label.unsqueeze(-1)

            # start_positions = start_positions * path_label
            # end_positions = end_positions * path_label


            # start/end에 대해 loss 계산

            start_logits = start_logits.reshape(-1, start_logits.size(-1))
            end_logits = end_logits.reshape(-1, end_logits.size(-1))
            start_positions = start_positions.reshape(-1,).long()
            end_positions = end_positions.reshape(-1,).long()

            start_loss = torch.sum(loss_fct(start_logits, start_positions).view(-1, batch_size), -1) / batch_size
            end_loss = torch.sum(loss_fct(end_logits, end_positions).view(-1, batch_size), -1) /batch_size

            # 최종 loss 계산
            total_loss = (start_loss + end_loss ) / 2 + hop_3_loss

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (total_loss,) + outputs

        return outputs # (loss), start_logits, end_logits
class ElectraForQuestionAnswering_graph(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_graph, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num
        self.max_dec_len = config.max_dec_len
        self.att_decoder = GraphAttentionDecoder(config.hidden_size)
        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size//2)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size//2)
        self.start_dense_2 = nn.Linear(in_features=config.hidden_size//2, out_features=config.hidden_size)
        self.end_dense_2 = nn.Linear(in_features=config.hidden_size//2, out_features=config.hidden_size)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # ELECTRA weight 초기화
        self.init_weights()

    def forward(
            self,
            input_ids=None,

            ##################
            attention_mask=None,
            token_type_ids=None,
            sent_masks=None,
            question_edge=None,
            context_edge=None,
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
        sentence_size = input_ids.size(1)

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_length, hidden_size]

        cls_output = sequence_output[:, 0, :]
        # cls_output : [batch, hidden]

        all_cls_output= torch.sum(cls_output, dim=0, keepdim=True) / batch_size
        # all_cls_output : [1, hidden]

        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]

        div_term = div_term.masked_fill(div_term==0, 1e-10)

        sentence_representation = sentence_masks.bmm(sequence_output)
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]

        sentence_representation = sentence_representation.reshape(batch_size*self.max_sent_num, 1, self.hidden_size)
        sentence_representation = sentence_representation.transpose(0, 1)
        # sentence_represntation : [1, batch*sent_num, hidden]

        context_edges = F.one_hot(context_edge, num_classes=batch_size * self.max_sent_num)
        context_edges = torch.sum(context_edges, 2)
        context_edges[:, :, 0] = 0
        # batch, max_sent, 20, 400

        last_hidden = None
        init_decoder_input = all_cls_output.unsqueeze(1)
        decoder_inputs = init_decoder_input
        # decoder_inputs : [batch(1), seq_len(1), hidden]
        evidence_list = []
        init_attention_mask = question_edge.view(1, 1, batch_size*self.max_sent_num)
        all_attention_masks = context_edges.view(batch_size*self.max_sent_num, batch_size*self.max_sent_num)
        attention_masks = init_attention_mask
        for dec_step in range(self.max_dec_len):
            decoder_inputs, last_hidden, evidence = self.att_decoder(last_hidden, decoder_inputs, sentence_representation, attention_masks)
            attention_masks=all_attention_masks[evidence[0]].reshape(1, 1, batch_size*self.max_sent_num)
            if torch.sum(attention_mask) == 0:
                attention_mask = init_attention_mask
            evidence_list.append(evidence)
        evidence_list = torch.stack(evidence_list, 0).transpose(0, 1)
        # last_hidden : (1, batch, hidden) ==> (batch, hidden, 1)
        evidence_list = evidence_list.expand(batch_size, -1)
        last_hidden = last_hidden.permute(1, 2, 0)

        start_representation = self.start_dense(sequence_output)
        end_representation = self.end_dense(sequence_output)


        # evidences : (batch, seq, 1)


        sequence_output_2 = torch.cat([start_representation, end_representation], -1)

        # logits : [batch_size, max_length, 2]
        logits = self.qa_outputs(sequence_output_2)

        # start_logits : [batch_size, max_length, 1]
        # end_logits : [batch_size, max_lenght, 1]
        start_logits, end_logits = logits.split(1, dim=-1)

        dense_start_sequence_outputs = self.start_dense_2(start_representation)
        dense_end_sequence_outputs = self.end_dense_2(end_representation)
        # dense_sequence_outputs : (batch, seq, hidden)
        last_hidden = last_hidden.expand(batch_size, -1, -1)
        start_evidences = dense_start_sequence_outputs.bmm(last_hidden)
        end_evidences = dense_end_sequence_outputs.bmm(last_hidden)
        start_logits +=start_evidences
        end_logits += end_evidences

        # start_logits : [batch_size, max_length]
        # end_logits : [batch_size, max_lenght]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # outputs = (start_logits, end_logits)
        outputs = (start_logits, end_logits, evidence_list) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언
            loss_fct = CrossEntropyLoss()

            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 최종 loss 계산
            total_loss = (start_loss + end_loss ) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (total_loss,) + outputs

        return outputs # (loss), start_logits, end_logits
