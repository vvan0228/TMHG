from random import random

from src.utils import WordPair
import os
import re
import json

import numpy as np

from collections import defaultdict
from itertools import accumulate
from transformers import AutoTokenizer
from typing import List, Dict
from loguru import logger
from tqdm import tqdm


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        config.mask_id = self.tokenizer.mask_token_id
        config.cls_id = self.tokenizer.cls_token_id
        config.pad_id = self.tokenizer.pad_token_id
        self.wordpair = WordPair()
        self.entity_dict = self.wordpair.entity_dic


    def get_dict(self):
        self.polarity_dict = self.config.polarity_dict

        # eg. BIESO {B-aspect:0, I-aspect:1 ...}
        self.aspect_dict = {}
        for w in self.config.bio_mode:
            self.aspect_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.asp_type)] = len(self.aspect_dict)

        self.target_dict = {}
        for w in self.config.bio_mode:
            self.target_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.tgt_type)] = len(self.target_dict)

        self.opinion_dict = {'O': 0}
        for p in self.polarity_dict:
            if p == 'O': continue
            for w in self.config.bio_mode[1:]:
                self.opinion_dict['{}-{}_{}'.format(w, self.config.opi_type, p)] = len(self.opinion_dict)

        self.relation_dict = {'O': 0, 'h2h': 1, 't2t': 2}

        return self.polarity_dict, self.target_dict, self.aspect_dict, self.opinion_dict, self.entity_dict, self.relation_dict

    def get_neighbor(self, utterance_spans, replies, max_length, speaker_ids, thread_nums):
        # utterance_mask = np.zeros([max_length, max_length], dtype=int)
        reply_mask = np.eye(max_length, dtype=int)
        for i, w in enumerate(replies):
            s1, e1 = utterance_spans[i]
            s0, e0 = utterance_spans[w + (1 if w == -1 else 0)]
            reply_mask[s0: e0 + 1, s1: e1 + 1] = 1
            reply_mask[s1: e1 + 1, s0: e0 + 1] = 1
            reply_mask[s0: e0 + 1, s0: e0 + 1] = 1
            reply_mask[s1: e1 + 1, s1: e1 + 1] = 1

        speaker_mask = np.zeros([max_length, max_length], dtype=int)
        for i, idx in enumerate(speaker_ids):
            # utterance_ids = [j for j, w in enumerate(speaker_ids) if w == idx]
            s0, e0 = utterance_spans[i]
            for j, idx1 in enumerate(speaker_ids):
                if idx != idx1: continue
                s1, e1 = utterance_spans[j]
                speaker_mask[s0: e0 + 1, s1: e1 + 1] = 1
                speaker_mask[s1: e1 + 1, s0: e0 + 1] = 1
                speaker_mask[s0: e0 + 1, s0: e0 + 1] = 1
                speaker_mask[s1: e1 + 1, s1: e1 + 1] = 1

        thread_mask = np.eye(max_length, dtype=int)
        thread_ends = accumulate(thread_nums)
        thread_spans = [(w - z, w) for w, z in zip(thread_ends, thread_nums)]
        for i, (s, e) in enumerate(thread_spans):
            if i == 0: continue
            head_start, head_end = utterance_spans[0]
            thread_mask[head_start: head_end + 1, head_start: head_end + 1] = 1
            for j in range(s, e):
                s0, e0 = utterance_spans[j]
                thread_mask[s0:e0 + 1, head_start:head_end + 1] = 1
                thread_mask[head_start:head_end + 1, s0:e0 + 1] = 1
                for k in range(s, e):
                    s1, e1 = utterance_spans[k]
                    thread_mask[s0 + 1: e0, s1 + 1: e1] = 1
                    thread_mask[s1 + 1: e1, s1 + 1: e1] = 1
                    thread_mask[s0 + 1: e0, s0 + 1: e0] = 1
                    thread_mask[s1 + 1: e1, s0 + 1: e0] = 1

        return reply_mask.tolist(), speaker_mask.tolist(), thread_mask.tolist()

    def get_utterance_topic_speaker_graphs(self, utterance_spans, replies, sentence_length, speakers, utterance_texts):
        graph = defaultdict(list)

        num_utterances = len(utterance_texts) 
        utt_nodes = list(range(num_utterances))

        # === 构建 utterance 回复边 (utt→utt) ===
        for cur_utt in range(num_utterances):
            if replies[cur_utt] != -1:
                graph['utt→utt'].append((replies[cur_utt], cur_utt, 1.0))

        # === 构建 topic 节点（只在深度≥3 的分支上创建） ===
        topic_nodes = []
        topic_node_texts = []

        utt2topic = {}

        for cur_utt in range(num_utterances):
            if replies[cur_utt] == -1:
                continue

            # 回溯当前分支
            prefix_utts = []
            current = cur_utt
            while current != -1:
                prefix_utts.append(current)
                current = replies[current]

            # 只有深度 ≥3 的分支才建 topic
            if len(prefix_utts) < 3:
                continue

            # 有序去重
            seen = set()
            unique_texts = []
            for u in prefix_utts:
                txt = utterance_texts[u]
                if txt not in seen:
                    seen.add(txt)
                    unique_texts.append(txt)

            topic_text = " [TOPIC] " + " ".join(unique_texts)

            topic_idx = len(utt_nodes) + len(topic_nodes)
            topic_nodes.append(topic_idx)
            topic_node_texts.append(topic_text)

            # 添加边：所有前序 utterance → topic → 当前 utterance
            for u in prefix_utts[:-1]:
                graph['utt→topic'].append((u, topic_idx, 1.0))
            graph['topic→utt'].append((topic_idx, cur_utt, 1.0))
            utt2topic[cur_utt] = topic_idx

            if len(topic_nodes) > 1:
                prev_topic = topic_nodes[-2]
                graph['topic→topic'].append((prev_topic, topic_idx, 1.0))

        prefix_groups = defaultdict(list)
        for cur_u, t_idx in utt2topic.items():
            path = []
            curr = replies[cur_u]
            while curr != -1:
                path.append(curr)
                curr = replies[curr]
            # 按照相同前缀序列和深度进行分组
            prefix_groups[tuple(path)].append(t_idx)

        # 对每个分组内的 topic 节点，两两相连
        for _, topics in prefix_groups.items():
            if len(topics) > 1:
                for i in range(len(topics)):
                    for j in range(i + 1, len(topics)):
                        t1, t2 = topics[i], topics[j]
                        graph['topic↔topic'].append((t1, t2, 1.0))
                        graph['topic↔topic'].append((t2, t1, 1.0))  

        # ===  构建speaker节点 ===
        speaker_nodes = {}
        for spk in set(speakers):
            spk_idx = len(utt_nodes) + len(topic_nodes) + len(speaker_nodes)
            speaker_nodes[spk] = spk_idx

        for idx, spk in enumerate(speakers):
            spk_node = speaker_nodes[spk]
            graph['speaker→utt'].append((spk_node, idx, 1.0))
            graph['utt→speaker'].append((idx, spk_node, 1.0))

        return {
            'nodes': {
                'utterance': utt_nodes,
                'topic': topic_nodes,
                'speaker': list(speaker_nodes.values())
            },
            'edges': {
                'utt→utt': graph.get('utt→utt', []),
                'utt→topic': graph.get('utt→topic', []),
                'topic→utt': graph.get('topic→utt', []),
                'topic→topic': graph.get('topic→topic', []),
                'speaker→utt': graph.get('speaker→utt', []),
                'utt→speaker': graph.get('utt→speaker', []),
                'topic↔topic': graph.get('topic↔topic', []) 
            },
            'node_texts': {
                'utterance': utterance_texts,
                'topic': topic_node_texts,
                'speaker': list(speaker_nodes.keys()) 
            }
        }

    def get_token_thread_graph(self, utterance_spans, replies, sentence_length, thread_range, tokens_text=None):
        from collections import defaultdict
        import numpy as np

        thread_graphs = defaultdict(lambda: {'tokens': [], 'utts': [], 'edges': {}})
        num_utterances = len(utterance_spans)
        depths = [0] * num_utterances

        for i in range(num_utterances):
            cur = i
            d = 0
            while replies[cur] != -1:
                d += 1
                cur = replies[cur]
            depths[i] = d

        for thread_id, (start, end) in enumerate(thread_range):
            thread_tokens = []
            thread_utts = list(range(start, end))
            edges = defaultdict(list)

            token2utt = {}
            weights = {}
            for utt_idx in range(start, end):
                s, e = utterance_spans[utt_idx]
                utt_tokens = list(range(s, e + 1))
                thread_tokens.extend(utt_tokens)
                for tok in utt_tokens:
                    token2utt[tok] = utt_idx
                    weights[tok] = 1.0 / (1 + depths[utt_idx])  # 越靠近根节点权重越大

            for tok, utt_idx in token2utt.items():
                edges['TOK–UTT'].append((tok, utt_idx, weights[tok]))
                edges['UTT–TOK'].append((utt_idx, tok, weights[tok]))

            for utt_idx in range(start, end):
                s, e = utterance_spans[utt_idx]
                utt_tokens = list(range(s, e + 1))
                for i in range(len(utt_tokens)):
                    for j in range(i + 1, len(utt_tokens)):
                        t1, t2 = utt_tokens[i], utt_tokens[j]
                        w = (weights[t1] + weights[t2]) / 2
                        edges['TOK–INTTOK'].append((t1, t2, w))
                        edges['TOK–INTTOK'].append((t2, t1, w))

            all_tokens = thread_tokens
            for i in range(len(all_tokens)):
                for j in range(i + 1, len(all_tokens)):
                    t1, t2 = all_tokens[i], all_tokens[j]
                    u1, u2 = token2utt[t1], token2utt[t2]
                    if u1 != u2:
                        w = (weights[t1] + weights[t2]) / 2
                        edges['TOK–OUTTOK'].append((t1, t2, w))
                        edges['TOK–OUTTOK'].append((t2, t1, w))

            thread_graphs[thread_id]['tokens'] = thread_tokens
            thread_graphs[thread_id]['utts'] = thread_utts
            thread_graphs[thread_id]['edges'] = edges

        return {
            'threads': thread_graphs,
            'node_texts': {
                tid: [f"token_{tok}" for tok in tg['tokens']] + [f"utt_{utt}" for utt in tg['utts']]
                for tid, tg in thread_graphs.items()
            }
        }


    def get_sentence_level_graph(self, utterance_texts, speakers, replies):
        graph = defaultdict(list)
        num_utts = len(utterance_texts)
        sent_offset = num_utts

        # 分句处理
        utt2sents = defaultdict(list)
        for idx, text in enumerate(utterance_texts):
            sentences = re.split(r'[。！？. ! ?]+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            for sent in sentences:
                sent_id = sent_offset + len(graph['sentence'])
                graph['sentence'].append(sent_id)
                utt2sents[idx].append(sent_id)

        # 构建节点
        graph['speaker'] = list(range(num_utts, num_utts + len(set(speakers))))
        graph['utterance'] = list(range(num_utts))
        graph['sentence'] = list(range(num_utts, num_utts + len(graph['sentence'])))

        # 边类型：speaker-utterance
        for idx, spk in enumerate(speakers):
            spk_node = num_utts + list(set(speakers)).index(spk)
            graph['speaker_utterance'].append((spk_node, idx, 1.0))
            graph['utterance_speaker'].append((idx, spk_node, 1.0))

        # 边类型：utterance-reply
        for idx, reply in enumerate(replies):
            if reply != -1:
                graph['utterance_reply'].append((idx, reply, 1.0))
                graph['utterance_reply'].append((reply, idx, 1.0))

        # 边类型：utterance-sentence
        for idx, sents in utt2sents.items():
            for sent in sents:
                graph['utterance_sentence'].append((idx, sent, 1.0))
                graph['sentence_utterance'].append((sent, idx, 1.0))

        # 1. 同一个 utterance 内部的句子之间连边
        for sents in utt2sents.values():
            for i in range(len(sents)):
                for j in range(i + 1, len(sents)):
                    graph['sentence_sentence'].append((sents[i], sents[j], 1.0))
                    graph['sentence_sentence'].append((sents[j], sents[i], 1.0))

        # 2. 只有具有明确 Reply 关系的 utterance 之间的句子才连边
        for idx, reply in enumerate(replies):
            if reply != -1:
                for sent_i in utt2sents[idx]:
                    for sent_j in utt2sents[reply]:
                        graph['sentence_sentence'].append((sent_i, sent_j, 0.7))
                        graph['sentence_sentence'].append((sent_j, sent_i, 0.7))

        return {
            'nodes': {'utterance': graph['utterance'], 'sentence': graph['sentence'], 'speaker': graph['speaker']},
            'edges': {
                'speaker_utterance': graph.get('speaker_utterance', []),
                'utterance_speaker': graph.get('utterance_speaker', []),
                'utterance_reply': graph.get('utterance_reply', []),
                'utterance_sentence': graph.get('utterance_sentence', []),
                'sentence_utterance': graph.get('sentence_utterance', []),
                'sentence_sentence': graph.get('sentence_sentence', [])
            },
            'node_texts': {'utterance': utterance_texts, 'sentence': [f"sent_{i}" for i in graph['sentence']], 'speaker': list(set(speakers))}
        }

    def convert_heterogeneous_graph_to_adjacency(self, graph_data, utterance_spans, max_seq_len):
        """
        将异构图转换为同构邻接矩阵，保持原有的图语义
        """
        # 初始化邻接矩阵
        adj_matrix = np.eye(max_seq_len, dtype=np.float32)

        # 1. 处理utterance节点之间的连接
        utt_edges = graph_data['edges'].get('utt→utt', [])
        for src_utt, dst_utt, weight in utt_edges:
            if src_utt < len(utterance_spans) and dst_utt < len(utterance_spans):
                src_start, src_end = utterance_spans[src_utt]
                dst_start, dst_end = utterance_spans[dst_utt]

                # utterance间的token连接
                for i in range(src_start, min(src_end + 1, max_seq_len)):
                    for j in range(dst_start, min(dst_end + 1, max_seq_len)):
                        adj_matrix[i][j] = max(adj_matrix[i][j], weight * 0.8)
                        adj_matrix[j][i] = max(adj_matrix[j][i], weight * 0.8)

        topic_to_utt = graph_data['edges'].get('topic→utt', [])
        utt_to_topic = graph_data['edges'].get('utt→topic', [])

        # 构建topic影响的utterance集合
        topic_influence = defaultdict(set)
        for topic_node, utt_node, weight in topic_to_utt:
            topic_influence[topic_node].add(utt_node)

        for utt_node, topic_node, weight in utt_to_topic:
            topic_influence[topic_node].add(utt_node)

        for topic_node, influenced_utts in topic_influence.items():
            influenced_utts = list(influenced_utts)
            for i in range(len(influenced_utts)):
                for j in range(i + 1, len(influenced_utts)):
                    utt_i, utt_j = influenced_utts[i], influenced_utts[j]
                    if utt_i < len(utterance_spans) and utt_j < len(utterance_spans):
                        start_i, end_i = utterance_spans[utt_i]
                        start_j, end_j = utterance_spans[utt_j]

                        for ti in range(start_i, min(end_i + 1, max_seq_len)):
                            for tj in range(start_j, min(end_j + 1, max_seq_len)):
                                adj_matrix[ti][tj] = max(adj_matrix[ti][tj], 0.6) 
                                adj_matrix[tj][ti] = max(adj_matrix[tj][ti], 0.6)

        for edge_name in ['topic→topic', 'topic↔topic']:
            for t1, t2, _ in graph_data['edges'].get(edge_name, []):
                utts_1 = list(topic_influence[t1])
                utts_2 = list(topic_influence[t2])
                for u1 in utts_1:
                    for u2 in utts_2:
                        if u1 < len(utterance_spans) and u2 < len(utterance_spans):
                            start_1, end_1 = utterance_spans[u1]
                            start_2, end_2 = utterance_spans[u2]
                            for ti in range(start_1, min(end_1 + 1, max_seq_len)):
                                for tj in range(start_2, min(end_2 + 1, max_seq_len)):
                                    adj_matrix[ti][tj] = max(adj_matrix[ti][tj], 0.6)
                                    adj_matrix[tj][ti] = max(adj_matrix[tj][ti], 0.6)

        # 3. 处理speaker连接
        speaker_to_utt = graph_data['edges'].get('speaker→utt', [])
        utt_to_speaker = graph_data['edges'].get('utt→speaker', [])

        # 构建同speaker的utterance集合
        speaker_utts = defaultdict(set)
        for speaker_node, utt_node, weight in speaker_to_utt:
            speaker_utts[speaker_node].add(utt_node)
        for utt_node, speaker_node, weight in utt_to_speaker:
            speaker_utts[speaker_node].add(utt_node)

        # 同speaker的utterance之间连接
        for speaker_node, utts in speaker_utts.items():
            utts = list(utts)
            for i in range(len(utts)):
                for j in range(i + 1, len(utts)):
                    utt_i, utt_j = utts[i], utts[j]
                    if utt_i < len(utterance_spans) and utt_j < len(utterance_spans):
                        start_i, end_i = utterance_spans[utt_i]
                        start_j, end_j = utterance_spans[utt_j]

                        # speaker身份连接
                        for ti in range(start_i, min(end_i + 1, max_seq_len)):
                            for tj in range(start_j, min(end_j + 1, max_seq_len)):
                                adj_matrix[ti][tj] = max(adj_matrix[ti][tj], 0.4)  # speaker连接权重
                                adj_matrix[tj][ti] = max(adj_matrix[tj][ti], 0.4)

        return adj_matrix

    def convert_token_thread_graph_to_adjacency(self, graph_data, max_seq_len):
        """
        将Token–Thread 图转换为邻接矩阵。
        支持以下边类型：
            - TOK–UTT / UTT–TOK
            - TOK–INTTOK
            - TOK–OUTTOK
        Utterance 节点的索引统一映射到 max_seq_len 之后的“扩展区”，
        但最终邻接矩阵仍保持 max_seq_len 大小，只保留 token 对 token 的影响。
        """
        import numpy as np

        adj_matrix = np.eye(max_seq_len, dtype=np.float32)

        for thread_id, thread_info in graph_data['threads'].items():
            edges = thread_info.get('edges', {})

            for edge_type, edge_list in edges.items():
                for (src, tgt, w) in edge_list:
                    if src < max_seq_len and tgt < max_seq_len:
                        adj_matrix[src][tgt] += float(w)
                        adj_matrix[tgt][src] += float(w)


        return adj_matrix.astype(np.float32)

    def convert_sentence_graph_to_adjacency(self, graph_data, utterance_spans, max_seq_len):
        """
        将句子级图转换为token级邻接矩阵
        """
        adj_matrix = np.eye(max_seq_len, dtype=np.float32)

        # 处理utterance-reply连接
        reply_edges = graph_data['edges'].get('utterance_reply', [])
        for utt_i, utt_j, weight in reply_edges:
            if utt_i < len(utterance_spans) and utt_j < len(utterance_spans):
                start_i, end_i = utterance_spans[utt_i]
                start_j, end_j = utterance_spans[utt_j]

                for ti in range(start_i, min(end_i + 1, max_seq_len)):
                    for tj in range(start_j, min(end_j + 1, max_seq_len)):
                        adj_matrix[ti][tj] = weight * 0.7
                        adj_matrix[tj][ti] = weight * 0.7

        # 处理speaker-utterance连接
        speaker_utt_edges = graph_data['edges'].get('speaker_utterance', [])
        # 按speaker分组utterance
        speaker_groups = defaultdict(list)
        for speaker_node, utt_node, weight in speaker_utt_edges:
            speaker_groups[speaker_node].append(utt_node)

        # 同speaker的utterance之间连接
        for speaker, utts in speaker_groups.items():
            for i in range(len(utts)):
                for j in range(i + 1, len(utts)):
                    utt_i, utt_j = utts[i], utts[j]
                    if utt_i < len(utterance_spans) and utt_j < len(utterance_spans):
                        start_i, end_i = utterance_spans[utt_i]
                        start_j, end_j = utterance_spans[utt_j]

                        for ti in range(start_i, min(end_i + 1, max_seq_len)):
                            for tj in range(start_j, min(end_j + 1, max_seq_len)):
                                adj_matrix[ti][tj] = max(adj_matrix[ti][tj], 0.5)
                                adj_matrix[tj][ti] = max(adj_matrix[tj][ti], 0.5)

        return adj_matrix

    def find_utterance_index(self, replies, sentence_lengths):
        utterance_collections = [i for i, w in enumerate(replies) if w == 0]
        # zero_index = utterance_collections[1]
        # for i in range(len(replies)):
        #     if i < zero_index: continue
        #     if replies[i] == 0:
        #         zero_index = i
        #     replies[i] = (i - zero_index)

        # 从第二个对话开始计算
        # 添加边界检查：这段代码确实存在潜在的索引越界风险。DiaASQ任务可能专注于多轮对话，单轮对话被过滤
        if len(utterance_collections) > 1:
            zero_index = utterance_collections[1]
            for i in range(len(replies)):
                if i < zero_index: continue
                if replies[i] == 0:
                    zero_index = i
                replies[i] = (i - zero_index)

        sentence_index = [w + 1 for w in replies]

        utterance_index = [[w] * z for w, z in zip(sentence_index, sentence_lengths)]
        utterance_index = [w for line in utterance_index for w in line]

        token_index = [list(range(sentence_lengths[0]))]
        lens = len(token_index[0])
        for i, w in enumerate(sentence_lengths):
            if i == 0: continue
            if sentence_index[i] == 1:
                distance = lens
            token_index += [list(range(distance, distance + w))]
            distance += w
        token_index = [w for line in token_index for w in line]

        utterance_collections = np.split(sentence_index, utterance_collections)

        thread_nums = list(map(len, utterance_collections))
        thread_ranges = [0] + list(accumulate(thread_nums))
        thread_lengths = [sum(sentence_lengths[thread_ranges[i]:thread_ranges[i + 1]]) for i in range(len(thread_ranges) - 1)]

        sent_idx2reply_idx = defaultdict(int)
        for sent_idx, reply in enumerate(replies):
            if reply == -1:
                sent_idx2reply_idx[sent_idx] = 0
            elif reply == 0:
                sent_idx2reply_idx[sent_idx] = 0
            else:
                sent_idx2reply_idx[sent_idx] = last_reply_idx
            last_reply_idx = sent_idx

        return utterance_index, token_index, thread_lengths, thread_nums, sent_idx2reply_idx

    def get_pair(self, full_triplets):
        pairs = {'ta': set(), 'ao': set(), 'to': set()}
        for i in range(len(full_triplets)):
            st, et, sa, ea, so, eo, p = full_triplets[i][:7]
            if st != -1 and sa != -1:
                pairs['ta'].add((st, et, sa, ea))

            if st != -1 and so != -1:
                pairs['to'].add((st, et, so, eo))

            if sa != -1 and eo != -1:
                pairs['ao'].add((sa, ea, so, eo))

        return pairs

    def transfer_polarity(self, pol):
        res = {'pos': 'pos', 'neg': 'neg'}
        return res.get(pol, 'other')

    def read_data(self, mode):
        """
        Read a JSON file, tokenize using BERT, and realign the indices of the original elements according to the tokenization results.
        """

        path = os.path.join(self.config.json_path, '{}.json'.format(mode))
        if self.config.testset_name is not None and 'test' in mode:
            path = os.path.join(self.config.json_path, '{}'.format(self.config.testset_name))
        print("dataset path: ", path)

        if not os.path.exists(path):
            raise FileNotFoundError('File {} not found! Please check your input and data path.'.format(path))

        content = json.load(open(path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_dialog = self.parse_dialogue(line, mode)
            res.append(new_dialog)
        return res

    def check_text(self, tokenized_text, source_text):
        if self.config.bert_path in ['roberta-large', 'roberta-base']:
            t0 = tokenized_text.lower()
            roberta_chars = 'â ī ¥ Ġ ð ł ĺ ħ Ł ŀ į Ŀ Į ĵ © ĵ ĳ ¶ ã'.split()
            unused = [self.config.unk, '##']
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                unused += roberta_chars
            for u in unused:
                t0 = t0.replace(u.lower(), '')
            t1 = source_text.replace(' ', '').lower()
            for k in self.config.unkown_tokens:
                t1 = t1.replace(k, '')
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                t1 = t1.replace('×', '').replace('≥', '')
            if t0 != t1:
                logger.info(t1 + '||' + t1)
                logger.info(tokenized_text + '||' + source_text)
                t2 = t0
                for u in unused:
                    t2 = t2.replace(u, '')
                raise AssertionError("--{}-- != --{}--".format(t0, t1))
            return t0 == t1

        t0 = tokenized_text.replace('##', '').replace(self.config.unk, '').lower()
        t1 = source_text.replace(' ', '').lower()
        for k in self.config.unkown_tokens:
            t1 = t1.replace(k, '')
        if t0 != t1:
            logger.info(t1 + '||' + t1)
            logger.info(tokenized_text + '||' + source_text)
            raise AssertionError("{} != {}".format(t0, t1))
        return t0 == t1

    def parse_dialogue(self, dialogue, mode):
        # get the list of sentences in the dialogue
        sentences = dialogue['sentences']

        # align_index_with_list: align the index of the original elements according to the tokenization results
        # eg. pieces2words = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4]
        if 'dep' in mode:
            piece_dep = dialogue['piece_dep']
            new_sentences = piece_dep['pieces']
            targets, aspects, opinions, triplets, pieces2words = [piece_dep[w] for w in['targets', 'aspects', 'opinions', 'triplets', 'dep_piece2ori_token']]
            # thread_piece = piece_dep['thread_pieces']
            # dialogue['thread_piece'] = thread_piece
        else:
            new_sentences, pieces2words = self.align_index_with_list(sentences)

            word2pieces = defaultdict(list)
            for p, w in enumerate(pieces2words):
                word2pieces[w].append(p)

            # get target, aspect and opinion respectively, and align to the new index

            # if 'train' not in mode and 'valid' not in mode:
            #     return dialogue
            targets, aspects, opinions = [dialogue[w] for w in ['targets', 'aspects', 'opinions']]
            targets = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z) for x, y, z in targets]
            aspects = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z) for x, y, z in aspects]
            opinions = [(word2pieces[x][0], word2pieces[y - 1][-1] + 1, z, self.transfer_polarity(w)) for x, y, z, w in
                        opinions]

            # polarity transfer and index transfer
            triplets = []
            for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in dialogue['triplets']:
                polarity = self.transfer_polarity(polarity)
                nts, nas, nos = [word2pieces[w][0] if w != -1 else -1 for w in [t_s, a_s, o_s]]
                nte, nae, noe = [word2pieces[w - 1][-1] + 1 if w != -1 else -1 for w in [t_e, a_e, o_e]]
                triplets.append((nts, nte, nas, nae, nos, noe, polarity, t_t, a_t, o_t))

        # Confirm the index again
        # Flatten the two-dimensional list and put the entire dialogue in a list
        news = [w for line in new_sentences for w in line]
        for ts, te, t_t in targets:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t in aspects:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t, _ in opinions:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:
            self.check_text(''.join(news[t_s:t_e]), t_t)
            self.check_text(''.join(news[a_s:a_e]), a_t) or a_s == -1
            if not self.check_text(''.join(news[o_s:o_e]), o_t) and o_s != -1:
                logger.info(''.join(news[o_s:o_e]) + '||' + o_t)
            self.check_text(''.join(news[o_s:o_e]), o_t) or o_s == -1

        # Put the elements into the dialogue object after converting the elements to the new index
        dialogue['sentences'] = new_sentences
        dialogue['targets'], dialogue['aspects'], dialogue['opinions'] = targets, aspects, opinions
        dialogue['triplets'] = triplets
        dialogue['pieces2words'] = pieces2words
        # DO tokenized
        return dialogue

    def align_index_with_list(self, sentences):
        """_summary_
        align the index of the original elements according to the tokenization results
        Args:
            sentences (_type_): List<str>
            e.g., xiao mi 12x is my favorite
        """
        pieces2word = []
        word_num = 0
        all_pieces = []
        for sentence in sentences:
            sentence = sentence.split()
            tokens = [self.tokenizer.tokenize(w) for w in sentence]
            cur_line = []
            for token in tokens:
                for piece in token:
                    pieces2word.append(word_num)
                word_num += 1
                cur_line += token
            all_pieces.append(cur_line)

        return all_pieces, pieces2word

    def align_index_with_list_dep(self, ori_tokens):
        pieces2word = []
        word_num = 0
        tokens = [self.tokenizer.tokenize(w) for w in ori_tokens]
        for token in tokens:
            for piece in token:
                pieces2word.append(word_num)
            word_num += 1

        return pieces2word

    def align_index(self, sentences):
        res, char2token = [], {}
        source_lens, token_lens = 0, 0
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if self.config.bert_path in ['roberta-large', 'bert-base-uncased']:
                c2t, tokens = self.alignment_roberta(sentence, tokens)
            else:
                c2t, tokens = self.alignment(sentence, tokens)
            res.append(tokens)
            for k, v in c2t.items():
                char2token[k + source_lens] = v + token_lens
            source_lens, token_lens = source_lens + len(sentence) + 1, token_lens + len(tokens)

        return res, char2token

    def alignment(self, source_sequence, tokenized_sequence: List[str], align_type: str = 'one2many') -> Dict:
        """[summary]
        # this is a function that to align sequcences  that before tokenized and after.
        Parameters
        ----------
        source_sequence : [type]
            this is the original sequence, whose type either can be str or list
        tokenized_sequence : List[str]
            this is the tokenized sequcen, which is a list of tokens.
        index_type : str, optional, default: str
            this indicate whether source_sequence is str or list, by default 'str'
        align_type : str, optional, default: one2many
            there may be several kinds of tokenizer style,
            one2many: one word in source sequence can be split into multiple tokens
            many2one: many word in source sequence will be merged into one token
            many2many: both contains one2many and many2one in a sequence, this is the most complicated situation.

        useage:
        source_sequence = "Here, we investigate the structure and dissociation process of interfacial water"
        tokenized_sequence = ['here', ',', 'we', 'investigate', 'the', 'structure', 'and', 'di', '##sso', '##ciation', 'process', 'of', 'inter', '##fa', '##cial', 'water']
        char2token = alignment(source_sequence, tokenized_sequence)
        print(char2token)
        for c, t in char2token.items():
            print(source_sequence[c], tokenized_sequence[t])
        """
        char2token = {}
        if isinstance(source_sequence, str) and align_type == 'one2many':
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j])
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length] == cur_token:
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or \
                                    tokenized_sequence[j + 1] == self.config.unk:
                                break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i + lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j + 1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence

    def alignment_roberta(self, source_sequence, tokenized_sequence: List[str]) -> Dict:
        # For English dataset
        char2token = {}
        if isinstance(source_sequence, str):
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j].strip('Ġ'))
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length].lower() == cur_token.strip('Ġ').lower():
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or \
                                    tokenized_sequence[j + 1] == self.config.unk:
                                if tokenized_sequence[j + 1].strip('#')[0] == 'i' and j + 1 < len(
                                        tokenized_sequence) and len(tokenized_sequence[j + 1].strip()) > 1:
                                    if i + lens + 1 < len(source_sequence) and source_sequence[i + lens + 1] == \
                                            tokenized_sequence[j + 1].strip('#')[1]:
                                        break
                                else:
                                    break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i + lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j + 1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence

    def repack_unknow(self, source_sequence):
        '''
        # sentence='🍎12💩', Bert can't recognize two contiguous emojis, so it recognizes the whole as '[UNK]'
        # We need to manually split it, recognize the words that are not in the bert vocabulary as UNK,
        and let BERT re-segment the parts that can be recognized, such as numbers
        # The above example processing result is: ['[UNK]', '12', '[UNK]']
        '''
        lst = list(re.finditer('|'.join(self.config.unkown_tokens), source_sequence))
        start, i = 0, 0
        new_tokens = []
        while i < len(lst):
            s, e = lst[i].span()
            if start < s:
                token = self.tokenizer.tokenize(source_sequence[start:s])
                new_tokens += token
                start = s
            else:
                new_tokens.append(self.config.unk)
                start = e
            i += 1
        if start < len(source_sequence):
            token = self.tokenizer.tokenize(source_sequence[start:])
            new_tokens += token
        return new_tokens

    def merge_same_thread(self, input_ids, input_masks, input_segments, sentence_length, utterance_index,
                          thread_length):

        merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length = [], [], [], []

        start_idx = 0
        idx_pairs = []
        j = 0
        sentence_len = sentence_length
        for tl in thread_length:
            all_len = 0
            while (j < len(sentence_len)):
                all_len += sentence_len[j]
                j += 1
                if all_len == tl:
                    idx_pairs.append((start_idx, j))
                    start_idx = j
                    break
        merged_input_ids = [[a for i in range(start, end) for a in input_ids[i]] for (start, end) in idx_pairs]
        # non speaker position in thread
        nonspeaker_token_positions = []
        for (start, end) in idx_pairs:
            cur_thread_len = 0
            ns_p = []
            for i in range(start, end):
                ns_p.append([cur_thread_len + 3, cur_thread_len + len(input_ids[i])])
                cur_thread_len += len(input_ids[i])
            nonspeaker_token_positions.append(ns_p)
        root_n_sp = nonspeaker_token_positions[0][0]
        for i in range(1, len(nonspeaker_token_positions)):
            for j in range(len(nonspeaker_token_positions[i])):
                nonspeaker_token_positions[i][j] = [nonspeaker_token_positions[i][j][0] + root_n_sp[1],
                                                    nonspeaker_token_positions[i][j][1] + root_n_sp[1]]
            nonspeaker_token_positions[i].insert(0, root_n_sp)
        nonspeaker_token_positions.pop(0)

        if 'roberta' in self.config.bert_path:
            merged_input_segments = [[0 for i in range(start, end) for a in input_segments[i]] for (start, end) in
                                     idx_pairs]
        else:
            merged_input_segments = [[0 if i == start else 1 for i in range(start, end) for a in input_segments[i]] for
                                     (start, end) in idx_pairs]
        merged_input_masks = [[a for i in range(start, end) for a in input_masks[i]] for (start, end) in idx_pairs]
        merged_sentence_length = [sum([sentence_length[i] for i in range(start, end)]) for (start, end) in idx_pairs]

        if self.config.root_merge == 1:
            root_merged_input_ids = [merged_input_ids[0] + merged_input_ids[i] for i in range(1, len(merged_input_ids))]
            root_merged_input_masks = [merged_input_masks[0] + merged_input_masks[i] for i in
                                       range(1, len(merged_input_masks))]
            root_merged_input_segments = [merged_input_segments[0] + merged_input_segments[i] for i in
                                          range(1, len(merged_input_segments))]
            root_merged_sentence_length = [merged_sentence_length[0] + merged_sentence_length[i] for i in
                                           range(1, len(merged_sentence_length))]
            return root_merged_input_ids, root_merged_input_masks, root_merged_input_segments, root_merged_sentence_length, nonspeaker_token_positions

        return merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, nonspeaker_token_positions

    def link_adj(self, adj_matrix, cls_list, sep_list, root_list, piece_list, head_list):
        # piece
        for pie in piece_list:
            for i in range(len(pie) - 1):
                adj_matrix[pie[i], pie[i + 1]] = 1.0
                adj_matrix[pie[i + 1], pie[i]] = 1.0

        # root
        for sent_idx in range(len(root_list)):
            for r_idx in range(len(root_list[sent_idx])):
                if r_idx + 1 < len(root_list[sent_idx]):
                    adj_matrix[root_list[sent_idx][r_idx][0], root_list[sent_idx][r_idx + 1][0]] = 1.0
                    adj_matrix[root_list[sent_idx][r_idx + 1][0], root_list[sent_idx][r_idx][0]] = 1.0

            if sent_idx + 1 < len(root_list):  # cross sentence
                adj_matrix[root_list[sent_idx][-1][0], root_list[sent_idx + 1][0][0]] = 1.0

        for i in range(len(sep_list)):
            adj_matrix[sep_list[i], cls_list[i]] = 1.0
            adj_matrix[cls_list[i], sep_list[i]] = 1.0

        return adj_matrix

    def get_adj_matrix(self, deprel, head_list):
        n = len(head_list)
        adj_matrix = np.array([[0.0] * n for i in range(n)])

        # 1. self-link edge
        for i in range(n):
            adj_matrix[i][i] = 1.0
        # 2. dependent edge
        for i in range(n):
            j = head_list[i]
            if j >= 0 and deprel[i] not in ['piece', 'ROOT', 'SENT_BEGIN', 'SENT_END']:
                adj_matrix[i][j] = 1.0
                adj_matrix[j][i] = 1.0  # make symmetric
        # 3. sentence edge
        cls_list, sep_list, root_list, piece_list = [], [], defaultdict(list), []
        utterance_index = 0
        i = 0
        while i < len(deprel):
            dep = deprel[i]
            if dep == 'SENT_BEGIN':
                cls_list.append(i)
            elif dep == 'SENT_END':
                sep_list.append(i)
                utterance_index += 1
            elif dep == 'ROOT':
                roots = []
                while i < len(deprel) and (deprel[i] == 'ROOT' or deprel[i] == 'piece'):
                    roots.append(i)
                    i += 1
                root_list[utterance_index].append(roots)
                continue
            elif dep == 'piece':
                pieces = [i - 1]
                while i < len(deprel) and deprel[i] == 'piece':
                    pieces.append(i)
                    i += 1
                piece_list.append(pieces)
                continue
            i += 1

        adj_matrix = self.link_adj(adj_matrix, cls_list, sep_list, root_list, piece_list, head_list)

        return adj_matrix

    def transform2indices(self, dataset, mode='train'):
        res = []
        for document in dataset:
            sentences, speakers, replies, pieces2words = [document[w] for w in
                                                          ['sentences', 'speakers', 'replies', 'pieces2words']]
            if 'train' in mode or 'valid' in mode:
                triplets, targets, aspects, opinions = [document[w] for w in
                                                        ['triplets', 'targets', 'aspects', 'opinions']]
            doc_id = document['doc_id']

            # sentence_length = list(map(lambda x : len(x) + 2, sentences))
            sentence_length = list(map(lambda x: len(x) + 2, sentences))

            # token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [w for line in token2sentid for w in line]  # [0, 0, 0, 1, 1, 2, 2, 2, 2, 2] 分别代表句子的id

            token2speaker = [[11] + [w] * len(z) + [10] for w, z in zip(speakers, sentences)]
            token2speaker = [w for line in token2speaker for w in
                             line]  # [11, 0, 0, 1, 1, 2, 2, 2, 2, 10] 11代表开始，10代表结束，其它代表说话者的id

            # New token indices (with CLS and SEP) to old token indices (without CLS and SEP)
            new2old = {}
            cur_len = 0
            for i in range(len(sentence_length)):
                for j in range(sentence_length[i]):
                    if j == 0 or j == sentence_length[i] - 1:
                        new2old[len(new2old)] = -1
                    else:
                        new2old[len(new2old)] = cur_len
                        cur_len += 1

            tokens = [[self.config.cls] + w + [self.config.sep] for s, w in zip(speakers, sentences)]

            # sentence_ids of each token (new token)
            nsentence_ids = [[i] * len(w) for i, w in enumerate(tokens)]
            nsentence_ids = [w for line in nsentence_ids for w in line]

            flatten_tokens = [w for line in tokens for w in line]
            sentence_end = [i - 1 for i, w in enumerate(flatten_tokens) if w == self.config.sep]
            sentence_start = [i + 1 for i, w in enumerate(flatten_tokens) if w == self.config.cls]
            # add speaker tokens at the end of each sentence
            for ts, s in zip(tokens, speakers):
                ts[-1] = self.tokenizer.tokenize(str(s))[0]

            utterance_texts = [_detok(pieces) for pieces in sentences]

            utterance_spans = list(zip(sentence_start, sentence_end))
            utterance_index, token_index, thread_length, thread_nums, sent_idx2reply_idx = self.find_utterance_index(
                replies, sentence_length)
            reply_mask, speaker_masks, thread_masks = self.get_neighbor(utterance_spans, replies, sum(sentence_length),
                                                                        speakers, thread_nums)

            thread_ends = list(accumulate(thread_nums))
            thread_range = [(0, thread_ends[0])]
            for i in range(1, len(thread_ends)):
                start = thread_ends[i - 1]
                end = thread_ends[i]
                thread_range.append((start, end))

            utt_topic_speaker_graph = self.get_utterance_topic_speaker_graphs(
                utterance_spans, replies, sentence_length, speakers, utterance_texts
            )
            token_thread_graph = self.get_token_thread_graph(
                utterance_spans, replies, sentence_length, thread_range
            )
            sentence_level_graph = self.get_sentence_level_graph(
                utterance_texts, speakers, replies
            )

            # 转换为邻接矩阵
            max_seq_len = sum(sentence_length)

            utt_topic_speaker_adj = self.convert_heterogeneous_graph_to_adjacency(
                utt_topic_speaker_graph, utterance_spans, max_seq_len
            )
            token_thread_adj = self.convert_token_thread_graph_to_adjacency(
                token_thread_graph, max_seq_len
            )
            sentence_level_adj = self.convert_sentence_graph_to_adjacency(
                sentence_level_graph, utterance_spans, max_seq_len
            )

            ex = {
                'utterance_spans': utterance_spans,
                'replies': replies,
                'sentence_length': sentence_length,
                'speakers': speakers,
                'utterance_texts': utterance_texts,
            }

            ex['graphs'] = {
                'utt_topic_speaker': utt_topic_speaker_graph,
                'token_thread': token_thread_graph,
                'sentence_level': sentence_level_graph
            }


            # add reply adj
            n = len(replies)
            utterance_level_reply_adj = np.array([[0.0] * n for i in range(n)])
            for i in range(len(replies)):
                utterance_level_reply_adj[i][i] = 1
                replied_idx = sent_idx2reply_idx[i]
                utterance_level_reply_adj[i][replied_idx] = 1
                utterance_level_reply_adj[replied_idx][i] = 1

            # DO speaker_adj
            utterance_level_speaker_adj = np.array([[0.0] * n for i in range(n)])
            for i in range(len(speakers)):
                cur_speaker = speakers[i]
                for j in range(len(speakers)):
                    if cur_speaker == speakers[j]:
                        utterance_level_speaker_adj[i][j] = 1.0

            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))

            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]

            if 'train' in mode or 'valid' in mode:
                targets = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in
                           targets]  # 2 * token2sentid[s] + 1 代表CLS+SEP
                aspects = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in aspects]
                opinions = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t, p in opinions]
                opinions = list(set(opinions))

                full_triplets, new_triplets = [], []
                # t_s-> target_start, t_e-> target_end, etc.
                for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:
                    new_index = lambda start, end: (-1, -1) if start == -1 else (
                    start + 2 * token2sentid[start] + 1, end + 2 * token2sentid[start])
                    t_s, t_e = new_index(t_s, t_e)
                    a_s, a_e = new_index(a_s, a_e)
                    o_s, o_e = new_index(o_s, o_e)
                    if polarity not in self.polarity_dict:
                        # 所有未知标签都归为 other
                        polarity = "other"
                    if polarity.lower() in ["ta", "to", "ao"]:
                        polarity = polarity.upper()
                    line = (t_s, t_e, a_s, a_e, o_s, o_e, self.polarity_dict[polarity])
                    full_triplets.append(line)
                    if all(w != -1 for w in [t_s, a_s, o_s]):
                        new_triplets.append(line)  # with CLS+SEP
                # 1 relation
                relation_lists = self.wordpair.encode_relation(full_triplets)  # eg (target head, aspect head, h2h)
                pairs = self.get_pair(
                    full_triplets)  # ta,to,ao full, eg (target_start, target_end, aspect_start, aspect_end)
                # 2 entity
                target_lists = self.wordpair.encode_entity(targets, 'ENT-T')
                aspect_lists = self.wordpair.encode_entity(aspects, 'ENT-A')
                opinion_lists = self.wordpair.encode_entity(opinions, 'ENT-O')
                entity_lists = target_lists + aspect_lists + opinion_lists  # eg. (target_start, target_end, 'ENT-T')
                # 3 polarity
                polarity_lists = self.wordpair.encode_polarity(new_triplets)  # eg（target head, opinion head, polarity)
            else:
                new_triplets, pairs, entity_lists, relation_lists, polarity_lists = [], [], [], [], []

            # DO merge_same_thread
            merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, nonspeaker_token_positions = \
                self.merge_same_thread(input_ids, input_masks, input_segments, sentence_length, utterance_index,
                                       thread_length)
            thread_range = [0] + list(accumulate(thread_nums))
            thread_range = [(thread_range[i], thread_range[i + 1]) for i in range(len(thread_range) - 1)]
            thread_idxes = [[i for i in range(start, end)] for (start, end) in thread_range][1:]
            thread_idxes = [[0] + w for w in thread_idxes]




            if 'dep' in mode:
                piece_dep = document['piece_dep']

                if self.config.merged_thread == 0:
                    deprels, heads = [piece_dep[w] for w in ['deprels', 'heads']]
                    adj_matrixes = [self.get_adj_matrix(d, h) for d, h in zip(deprels, heads, )]
                    assert len(deprels) == len(heads)
                else:
                    thread_deprels, thread_heads, = [piece_dep[w] for w in ['thread_deprels', 'thread_heads', ]]
                    adj_matrixes = [self.get_adj_matrix(d, h) for d, h in zip(thread_deprels, thread_heads, )]
                    assert len(thread_deprels) == len(thread_heads)


                res.append((doc_id, speakers, input_ids, input_masks, input_segments, sentence_length, nsentence_ids,
                            utterance_index, token_index,
                            thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words,
                            new2old,
                            new_triplets, pairs, entity_lists, relation_lists, polarity_lists, thread_idxes,
                            merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length,
                            adj_matrixes, utterance_level_reply_adj, utterance_level_speaker_adj,
                            utt_topic_speaker_adj, token_thread_adj, sentence_level_adj
                            ))

            else:

                res.append((doc_id, speakers, input_ids, input_masks, input_segments, sentence_length, nsentence_ids,
                            utterance_index, token_index,
                            thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words,
                            new2old,
                            new_triplets, pairs, entity_lists, relation_lists, polarity_lists, thread_idxes,
                            merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length,
                            utt_topic_speaker_adj, token_thread_adj, sentence_level_adj
                            ))

        return res

    def forward(self):
        # modes default: 'train valid test'
        modes = self.config.input_files
        datasets = {}

        for mode in modes.split():
            data = self.read_data(mode)  # have been tokenized and aligned
            datasets[mode] = data

        label_dict = self.get_dict()

        res = {}
        for mode in modes.split():
            res[mode] = self.transform2indices(datasets[mode], mode)

        res['label_dict'] = label_dict
        return res