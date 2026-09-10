#!/usr/bin/env python
# _*_ coding:utf-8 _*_


from src.Roberta import MultiHeadAttention, InteractionAttention
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from itertools import accumulate

import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, d_out=None):
        super(PositionwiseFeedForward, self).__init__()
        if d_out is None: d_out = d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class InteractLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, config=None):
        super(InteractLayer, self).__init__()
        head_size = int(d_model / num_heads)
        self.config = config
        self.interactionAttention = InteractionAttention(num_heads, d_model, head_size, head_size, dropout,
                                                         config=config)

        self.layer_norm_pre = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layer_norm_post = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_x, mask, sentence_length, ):
        x = self.layer_norm_pre(self.interactionAttention(x, global_x, mask, )[0] + x)
        x = self.layer_norm_post(self.ffn(x) + x)
        x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, config, layer_num, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.config = config
        self.layer_list = nn.ModuleList()
        for i in range(layer_num):
            if i == layer_num - 1:
                self.layer_list.append(nn.Linear(hidden_dim, output_dim))
            elif i == 0:
                self.layer_list.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layer_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.gnn_dropout = nn.Dropout(dropout)
        self.gnn_activation = F.gelu

    def forward(self, x, mask, adj):
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=x.dtype, device=x.device)
        else:
            adj = adj.to(x.device, dtype=x.dtype)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        else:
            mask = mask.to(x.device, dtype=x.dtype)

        # normalize dims: adj -> [B, L, L]; mask -> [B, L]
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1).contiguous()

        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(x.size(0), -1).contiguous()

        # if batch mismatch and adj.batchsize==1, expand; else raise informative error
        if adj.size(0) != x.size(0):
            if adj.size(0) == 1:
                adj = adj.expand(x.size(0), -1, -1).contiguous()
            else:
                raise RuntimeError(f"[GCN] batch mismatch: adj.batch={adj.size(0)} vs x.batch={x.size(0)}")

        # if seq_len mismatch, pad/crop adj to x.size(1)
        if adj.size(1) != x.size(1):
            new = adj.new_zeros(adj.size(0), x.size(1), x.size(1))
            s = min(adj.size(1), x.size(1))
            new[:, :s, :s] = adj[:, :s, :s]
            adj = new

        D_hat = torch.diag_embed(torch.pow(torch.sum(adj, dim=-1), -1))
        if torch.isinf(D_hat).any():
            D_hat[torch.isinf(D_hat)] = 0.0
        adj = torch.matmul(D_hat, adj)

        x_mask = mask.unsqueeze(-1)  # .expand(-1, -1, x.size(-1))
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                x = self.gnn_dropout(x)
            x = torch.matmul(x, layer.weight.T) + layer.bias
            x = torch.matmul(adj, x)
            x = x * x_mask
            x = self.gnn_activation(x)

        return x

class StructureAwareAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.graph_weights = nn.Parameter(torch.ones(3))

    def forward(self, x, adj_list, mask):
        B, L, H = x.shape

        q = self.q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        structural_bias = torch.zeros((B, L, L), device=x.device)
        for i, adj in enumerate(adj_list):
            if adj is not None:
     
                structural_bias = structural_bias + self.graph_weights[i] * adj

        scores = scores + structural_bias.unsqueeze(1)

        # 4. Padding Mask (-1e9)
        if mask is not None:
            pad_mask = (mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(pad_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, H)
        context = self.out(context)

        return self.layer_norm(x + self.dropout(context))


class BertWordPair(nn.Module):
    def __init__(self, config):
        super(BertWordPair, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.bert_path)

        bert_config = AutoConfig.from_pretrained(config.bert_path)
        bh = bert_config.hidden_size
        nhead = bert_config.num_attention_heads
        att_head_size = int(bh / nhead)

        self.config.loss_weight = {'ent': int(self.config.loss_w[0]), 'rel': int(self.config.loss_w[1]),
                                   'pol': int(self.config.loss_w[2])}

        self.inner_dim = 256
        self.ent_dim = self.inner_dim * 4 * 4
        self.rel_dim = self.inner_dim * 4 * 3
        self.pol_dim = self.inner_dim * 4 * 4

        self.dense_all = nn.Linear(bert_config.hidden_size, self.ent_dim + self.rel_dim + self.pol_dim)

        self.dropout = nn.Dropout(config.dropout)

        self.interactLayer = InteractLayer(
            bert_config.hidden_size,
            bert_config.num_attention_heads,
            bert_config.hidden_dropout_prob,
            config
        )

        self.layernorm = nn.LayerNorm(bh, eps=1e-12)
        self.syngcn = GCN(config, config.gnn_layer_num, bh, bh, bh, config.gnn_dropout)

        self.semgcn = GCN(config, config.gnn_layer_num, bh, bh, bh, config.gnn_dropout)
        self.semantic_attention = MultiHeadAttention(bert_config.num_attention_heads, bh, att_head_size, att_head_size,
                                                     bert_config.attention_probs_dropout_prob)

        self.structure_aware_attn = StructureAwareAttention(
            hidden_size=bh,
            num_heads=bert_config.num_attention_heads,
            dropout=config.gnn_dropout
        )

        self.utt_topic_speaker_gcn = GCN(config, config.extra_gnn_layer_num, bh, bh, bh, config.gnn_dropout)
        self.token_thread_gcn = GCN(config, config.extra_gnn_layer_num, bh, bh, bh, config.gnn_dropout)
        self.sentence_level_gcn = GCN(config, config.extra_gnn_layer_num, bh, bh, bh, config.gnn_dropout)

        max_concat_dim = bh * 4 
        self.linear_proj = nn.Linear(max_concat_dim, bh)

        self.use_utt_topic_speaker_graph = getattr(config, 'use_utt_topic_speaker_graph', True)
        self.use_token_thread_graph = getattr(config, 'use_token_thread_graph', True)
        self.use_sentence_level_graph = getattr(config, 'use_sentence_level_graph', True)

        self.gate_utt_topic_speaker = nn.Linear(bh * 2, 1)
        self.gate_token_thread = nn.Linear(bh * 2, 1)
        self.gate_sentence_level = nn.Linear(bh * 2, 1)

        # topk
        self.topK_select_layer = nn.Linear(bh, 1)
        self.utt_linear = nn.Linear(3 * bh, bh)

        self.dscgcn = GCN(config, config.dscgnn_layer_num, bh, bh, bh, config.gnn_dropout)
        self.global_layernorm = nn.LayerNorm(bh, eps=1e-12)

    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        https://blog.csdn.net/weixin_43646592/article/details/130924280
        """
        output_dim = self.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.config.device)  # 128
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices  # [seq_len, 128]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq_len, 128, 2]
        embeddings = embeddings.repeat((1, *([1] * len(embeddings.shape))))  # [1, seq_len, 128, 2]
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))  # [1, seq_len, 256]
        embeddings = embeddings.squeeze(0)
        return embeddings

    def get_instance_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index, thread_length, pos_type):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        token_index : 相对根的线程token位置
        """

        seq_len, num_classes = qw.shape[:2]

        accu_index = [0] + list(accumulate(thread_length))

        logits = qw.new_zeros([seq_len, seq_len, num_classes])

        # Compute the ROPE matrix
        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i + 1]
                cstart, cend = accu_index[j], accu_index[j + 1]

                cur_qw, cur_kw = qw[rstart:rend], kw[cstart:cend]
                x, y = token_index[rstart:rend], token_index[cstart:cend]

                x = - x if i > 0 and i < j else x
                y = - y if j > 0 and i > j else y

                x_pos_emb = self.custom_sinusoidal_position_embedding(x, pos_type)  # 38，256（38是第一个句子的长度
                y_pos_emb = self.custom_sinusoidal_position_embedding(y, pos_type)

                x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)  # 38， 1， 256
                x_sin_pos = x_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_qw2 = torch.stack([-cur_qw[..., 1::2], cur_qw[..., ::2]], -1)  # [38, 6, 128, 2]
                cur_qw2 = cur_qw2.reshape(cur_qw.shape)  # [38, 6, 256]
                cur_qw = cur_qw * x_cos_pos + cur_qw2 * x_sin_pos  # [38, 6, 256]

                y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                y_sin_pos = y_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_kw2 = torch.stack([-cur_kw[..., 1::2], cur_kw[..., ::2]], -1)
                cur_kw2 = cur_kw2.reshape(cur_kw.shape)
                cur_kw = cur_kw * y_cos_pos + cur_kw2 * y_sin_pos

                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw, cur_kw).contiguous()  # 38 38 6， 38 34 6
                logits[rstart:rend,
                cstart:cend] = pred_logits  # [0:38, 0:38]=[38,38,6] [0:38, 108:142]=[38,34,6] [38:108, 108:142]=[70,34,6]

        return logits

    def get_ro_embedding(self, qw, kw, token_index, thread_lengths, pos_type):
        # qw_res = qw.new_zeros(*qw.shape)
        # kw_res = kw.new_zeros(*kw.shape)
        # qw,kw (batch_size, seq_len, 6, 256)
        logits = []
        batch_size = qw.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding(qw[i], kw[i], token_index[i], thread_lengths[i],
                                                      pos_type)  # [seqlen, seqlen, classnums]
            logits.append(pred_logits)
        logits = torch.stack(logits)
        return logits

    def classify_matrix(self, kwargs, sequence_outputs, input_labels, masks, mat_name='ent'):

        utterance_index, token_index, thread_lengths = [kwargs[w] for w in
                                                        ['utterance_index', 'token_index', 'thread_lengths']]

        outputs = torch.split(sequence_outputs, self.inner_dim * 4, dim=-1)  # (batch_size, seq_len, 256*4) * 6
        outputs = torch.stack(outputs, dim=-2)  # (batch_size, seq_len, 6, 256*4)

        q_token, q_utterance, k_token, k_utterance = torch.split(outputs, self.inner_dim,
                                                                 dim=-1)  # (batch_size, seq_len, 6, 256)

        if self.config.use_rope == True:
            if mat_name == 'ent':
                # [batch_size, seq_len, seq_len, class_nums]
                pred_logits = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths,
                                                    pos_type=0)  # pos_type=0 for token-level relative distance encoding
            else:
                pred_logits0 = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths, pos_type=0)
                pred_logits1 = self.get_ro_embedding(q_utterance, k_utterance, utterance_index, thread_lengths,
                                                     pos_type=1)  # pos_type=1 for utterance-level relative distance encoding
                pred_logits = pred_logits0 + pred_logits1
        else:
            # without rope, use dot-product attention directly
            pred_logits = torch.einsum('bmhd,bnhd->bmnh', q_token, k_token).contiguous()

        nums = pred_logits.shape[-1]

        # alpha^k = loss weight
        criterion = nn.CrossEntropyLoss(
            sequence_outputs.new_tensor([1.0] + [self.config.loss_weight[mat_name]] * (nums - 1)))

        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]

        loss = criterion(active_logits, active_labels)

        return loss, pred_logits

    def merge_sentence(self, sequence_outputs, input_masks, dialogue_length):
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                stack.append(sequence_outputs[j, :lens])
            res.append(torch.cat(stack))
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res  # batch_size, max_dialogue_length, hidden_size

    def root_merge_sentence(self, sequence_outputs, input_masks, dialogue_length, thread_lengths):
        if self.config.root_merge == 0:
            return self.merge_sentence(sequence_outputs, input_masks, dialogue_length)

        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            root_stack = []
            root_len = thread_lengths[i][0]
            for j in range(s, e):
                lens = input_masks[j].sum()
                root_stack.append(sequence_outputs[j, :root_len])
                stack.append(sequence_outputs[j, root_len:lens])

            root = torch.stack(root_stack).sum(0) / len(root_stack)

            stack = [root] + stack
            res.append(torch.cat(stack))
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res  # batch_size, max_dialogue_length, hidden_size

    def topk_aggregate(self, sentence_sequence_outputs, global_masks):
        batch_size, max_dialogue_length, hidden_size = sentence_sequence_outputs.shape
        batch_size, max_sentence_num, max_dialogue_length, _ = global_masks.shape

        sentence_lengths = global_masks.sum(dim=2).squeeze(-1)

        split_sentences = []

        for i in range(batch_size):
            split_sentences.append([])
            for j in range(max_sentence_num):
                sentence_length = sentence_lengths[i, j]
                if sentence_length > 0:
                    start_index = (global_masks[i, j, :, :] == 1).nonzero()[0, 0].item()
                    end_index = int(start_index + sentence_length.item())
                    token_representation = sentence_sequence_outputs[i, start_index:end_index - 1, :]
                    speaker_representation = sentence_sequence_outputs[i, end_index - 1, :]
                    score = self.topK_select_layer(token_representation).squeeze(-1) / (sentence_length - 1)
                    # get topk of sentence: pooling[avg, max, speaker] as sentence representation
                    k = int(self.config.topk * sentence_length)
                    k = k if k > 0 else 1

                    topk = torch.topk(score, k, dim=0, largest=True)[1]
                    score = torch.softmax(score[topk], dim=0)
                    token_representation = token_representation[topk]

                    token_representation = token_representation * score.unsqueeze(-1)

                    utt_representation = self.utt_linear(torch.cat(
                        (token_representation.mean(dim=0), token_representation.max(dim=0)[0], speaker_representation),
                        dim=-1))

                    split_sentences[i].append(utt_representation)
                else:
                    split_sentences[i].append(sentence_sequence_outputs.new_zeros([hidden_size]))
        split_sentences = torch.stack([torch.stack(bat) for bat in split_sentences], dim=0)

        return split_sentences

    def global_encoding(self, speaker_ids, sentence_sequence_outputs, global_masks, utterance_level_reply_adj,
                        utterance_level_speaker_adj, utterance_level_mask):
        # sentence_sequence_outputs: batch_size, max_dialogue_length, hidden_size
        # global_masks: batch_size, max_sentence_num, max_dialogue_length, 1

        utterance_sequence = self.topk_aggregate(sentence_sequence_outputs, global_masks)
        global_outputs = self.dscgcn(utterance_sequence, utterance_level_mask, utterance_level_reply_adj)
        global_outputs = self.global_layernorm(utterance_sequence + global_outputs)

        return global_outputs

    def utterance2thread(self, sequence_outputs, thread_idxes, sentence_length, thread_lengths, merged_input_masks):
        # sequence_outputs: batch_size, max_sentence_length, hidden_size
        thread_num, max_thread_len = merged_input_masks.shape

        thread_sequence_output = sequence_outputs.new_zeros([thread_num, max_thread_len, sequence_outputs.shape[-1]])
        thread_idx = 0
        for bat_idx, bat in enumerate(thread_idxes):
            for t_idx, thread in enumerate(bat):
                thread_list = []
                for s_idx, sent_idx in enumerate(thread):
                    thread_list.append(sequence_outputs[bat_idx, :sentence_length[bat_idx][sent_idx], :])
                thread_list = torch.cat(thread_list, dim=0)
                thread_sequence_output[thread_idx, :thread_list.shape[0], :] = thread_list
                thread_idx += 1

        return thread_sequence_output

    def forward(self, **kwargs):
        if self.config.merged_thread == 0:
            input_ids, input_masks, input_segments = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments']]

        sentence_length, thread_idxes, merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, merged_dialog_length, thread_lengths, adj_matrixes \
            = [kwargs[w] for w in
               ['sentence_length', 'thread_idxes', 'merged_input_ids', 'merged_input_masks', 'merged_input_segments',
                'merged_sentence_length', 'merged_dialog_length', 'thread_lengths', 'adj_matrixes', ]]

        ent_matrix, rel_matrix, pol_matrix = [kwargs[w] for w in ['ent_matrix', 'rel_matrix', 'pol_matrix']]
        reply_masks, speaker_masks, thread_masks = [kwargs[w] for w in ['reply_masks', 'speaker_masks', 'thread_masks']]
        sentence_masks, full_masks, dialogue_length = [kwargs[w] for w in
                                                       ['sentence_masks', 'full_masks', 'dialogue_length']]

        utt_topic_speaker_graph = kwargs.get("utt_topic_speaker_graph", None)
        token_thread_graph = kwargs.get("token_thread_graph", None)
        sentence_level_graph = kwargs.get("sentence_level_graph", None)


        # DO
        # 1. bert encoding
        if self.config.merged_thread == 1:
            sequence_outputs = \
                self.bert(merged_input_ids, token_type_ids=merged_input_segments,
                          attention_mask=merged_input_masks)[
                    0]  # utterance_num, seq_len, hidden_size
            sentence_sequence_outputs = self.root_merge_sentence(sequence_outputs, merged_input_masks,
                                                                 merged_dialog_length, thread_lengths)
        else:  # w/o thread
            sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[
                0]  # utterance_num, seq_len, hidden_size
            sentence_sequence_outputs = self.merge_sentence(sequence_outputs, input_masks, dialogue_length)

        sentence_sequence_outputs = self.dropout(sentence_sequence_outputs)

        # 2. local encoding
        # 2.1 add thread gcn syntactic
        if self.config.merged_thread == 1:
            syngcn_outputs = self.syngcn(sequence_outputs, merged_input_masks, adj_matrixes)
            syngcn_outputs = self.root_merge_sentence(syngcn_outputs, merged_input_masks, merged_dialog_length,
                                                      thread_lengths)
        else:  # w/o thread
            syngcn_outputs = self.syngcn(sequence_outputs, input_masks, adj_matrixes)
            syngcn_outputs = self.merge_sentence(syngcn_outputs, input_masks, dialogue_length)

        syngcn_outputs = self.dropout(syngcn_outputs)

        # 2.2 add thread gcn semantic
        _, semantic_adj = self.semantic_attention(sequence_outputs, sequence_outputs, sequence_outputs)
        semantic_adj = semantic_adj.mean(dim=1)
        if self.config.merged_thread == 1:
            semgcn_output = self.semgcn(sequence_outputs, merged_input_masks, semantic_adj)
            semgcn_output = self.root_merge_sentence(semgcn_output, merged_input_masks, merged_dialog_length,
                                                     thread_lengths)
        else:  # w/o thread
            semgcn_output = self.semgcn(sequence_outputs, input_masks, semantic_adj)
            semgcn_output = self.merge_sentence(semgcn_output, input_masks, dialogue_length)
        semgcn_output = self.dropout(semgcn_output)

        # 2.3 integrate syntactic and semantic 
        local_context = sentence_sequence_outputs + syngcn_outputs + semgcn_output

        def _to_tensor_on_device(obj, ref_tensor):
            if obj is None: return None
            if not isinstance(obj, torch.Tensor): obj = torch.tensor(obj, dtype=ref_tensor.dtype)
            return obj.to(ref_tensor.device, dtype=ref_tensor.dtype)

        def _prepare_adj(adj, batch_size, target_len, ref_tensor):
            if adj is None: return None
            adj = _to_tensor_on_device(adj, ref_tensor)
            if adj.dim() == 2:
                L = adj.size(0)
                if L != target_len:
                    new = adj.new_zeros(target_len, target_len)
                    s = min(L, target_len)
                    new[:s, :s] = adj[:s, :s]
                    adj = new
                adj = adj.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            elif adj.dim() == 3:
                if adj.size(0) != batch_size:
                    adj = adj.expand(batch_size, -1, -1).contiguous() if adj.size(0) == 1 else adj
                if adj.size(1) != target_len:
                    new = adj.new_zeros(batch_size, target_len, target_len)
                    s = min(adj.size(1), target_len)
                    new[:, :s, :s] = adj[:, :s, :s]
                    adj = new
            return adj

        batch_size, seq_len = sentence_sequence_outputs.shape[:2]

        doc_mask = sentence_sequence_outputs.new_zeros(batch_size, seq_len)
        for i, l in enumerate(dialogue_length):
            doc_mask[i, :l] = 1.0

        adj_topic = _prepare_adj(utt_topic_speaker_graph, batch_size, seq_len, sentence_sequence_outputs)
        adj_thread = _prepare_adj(token_thread_graph, batch_size, seq_len, sentence_sequence_outputs)
        adj_sent = _prepare_adj(sentence_level_graph, batch_size, seq_len, sentence_sequence_outputs)

        adj_list = [adj_topic, adj_thread, adj_sent]

        concat_list = []

        if adj_topic is not None:
            feat = self.utt_topic_speaker_gcn(sentence_sequence_outputs, doc_mask, adj_topic)
            gate_val = torch.sigmoid(
                self.gate_utt_topic_speaker(torch.cat([sentence_sequence_outputs, feat], dim=-1)))
            concat_list.append(gate_val * feat)

        if adj_thread is not None:
            feat = self.token_thread_gcn(sentence_sequence_outputs, doc_mask, adj_thread)
            gate_val = torch.sigmoid(self.gate_token_thread(torch.cat([sentence_sequence_outputs, feat], dim=-1)))
            concat_list.append(gate_val * feat)

        if adj_sent is not None:
            feat = self.sentence_level_gcn(sentence_sequence_outputs, doc_mask, adj_sent)
            gate_val = torch.sigmoid(self.gate_sentence_level(torch.cat([sentence_sequence_outputs, feat], dim=-1)))
            concat_list.append(gate_val * feat)

        if len(concat_list) > 0:

            concat_list.append(sentence_sequence_outputs)
            fused_graph = torch.cat(concat_list, dim=-1)

            H_c = local_context + self.linear_proj(fused_graph)
        else:
            H_c = local_context

        sequence_outputs = self.structure_aware_attn(H_c, adj_list, doc_mask)

        # 3. global encoding
        global_masks, utterance_level_reply_adj, utterance_level_speaker_adj, utterance_level_mask, speaker_ids = [
            kwargs[w] for w in
            ['global_masks', 'utterance_level_reply_adj', 'utterance_level_speaker_adj', 'utterance_level_mask',
             'speaker_ids']]
        global_outputs = self.global_encoding(speaker_ids, sentence_sequence_outputs, global_masks,
                                              utterance_level_reply_adj, utterance_level_speaker_adj,
                                              utterance_level_mask)

        # 4. Interaction attention
        thread_masks = thread_masks.bool().unsqueeze(1)
        sequence_outputs = self.interactLayer(sequence_outputs, global_outputs, thread_masks,
                                              sentence_length=sentence_length, )

        # 5. decode
        sequence_outputs = self.dense_all(sequence_outputs)
        sequence_ent = sequence_outputs[:, :, :self.ent_dim]
        sequence_rel = sequence_outputs[:, :, self.ent_dim:self.ent_dim + self.rel_dim]
        sequence_pol = sequence_outputs[:, :, self.ent_dim + self.rel_dim:]

        ent_loss, ent_logit = self.classify_matrix(kwargs, sequence_ent, ent_matrix, sentence_masks, 'ent')
        rel_loss, rel_logit = self.classify_matrix(kwargs, sequence_rel, rel_matrix, full_masks, 'rel')
        pol_loss, pol_logit = self.classify_matrix(kwargs, sequence_pol, pol_matrix, full_masks, 'pol')

        total_loss = ent_loss + rel_loss + pol_loss

        return total_loss, [ent_loss, rel_loss, pol_loss], (ent_logit, rel_logit, pol_logit)