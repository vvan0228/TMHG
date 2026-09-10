#!/usr/bin/env python

import torch
import numpy as np
from attrdict import AttrDict
from scipy.linalg import block_diag
from collections import defaultdict

from itertools import accumulate
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import random
from loguru import logger
import json

from src.common import WordPair
from src.preprocess import Preprocessor
from src.run_eval import Template as Run_eval

from transformers import AutoTokenizer
import traceback
import requests
import re
import time



def request_llm_judge(llm_payload, dialogue_text, max_retries=2):

    api_key = ""
    base_url = ""

    system_prompt = """## Instruction ##
你是一个顶级的自然语言处理与情感分析专家。你的任务是审查对话情感四元组抽取（DiaASQ）任务中的候选预测。
请结合上下文多轮对话，在模型产生犹豫的候选类别中做出精准判定。

## Schema Definition ##
@dataclass
class NoneEntity:
    \"\"\"非实体词/背景词：不属于讨论目标、属性或评价的普通词语。\"\"\"
@dataclass
class Target:
    \"\"\"目标词：表示讨论的主体对象。例如：'屏幕'、'客服'。\"\"\"
@dataclass
class Aspect:
    \"\"\"方面词：指目标的具体属性特征。例如：'分辨率'、'态度'。\"\"\"
@dataclass
class Opinion:
    \"\"\"意见词：针对该方面所表达的态度或评价。例如：'非常清晰'、'太差了'。\"\"\"

## Evaluation Steps ##
小模型会在某些词的分类上产生犹豫。我会提供它预测的概率前三名（Option_1, Option_2, Option_3）。请结合整个对话上下文，判断哪一个类别最准确。

## Output Format ##
必须仅输出合法的 JSON 对象。在做出选择前，必须先在 "reason" 字段给出简短的判断理由。
{
  "corrections": {
    "ent_choice_0": {
        "reason": "在对话的上下文中，该词被用户用来评价手机属性，故为Aspect。",
        "choice": "option_2"
    }
  }
}

## Examples ##
Input Candidates:
[{"id": "ent_choice_ex1", "word": "画质", "option_1": {"type": "Aspect", "prob": 0.45}, "option_2": {"type": "Target", "prob": 0.40}, "option_3": {"type": "None", "prob": 0.15}}]
Output:
{
  "corrections": {
    "ent_choice_ex1": {
        "reason": "'画质'是电视的属性特征，而非讨论主体本身，故归为Aspect。",
        "choice": "option_1"
    }
  }
}
"""


    safe_dialogue = dialogue_text[:2500] + ("..." if len(dialogue_text) > 2500 else "")
    user_prompt = f"## Context ##\n{safe_dialogue}\n\n## Candidates ##\n{json.dumps(llm_payload, ensure_ascii=False)}"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048,  # 明确告诉服务器允许输出长文本
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            no_proxies = {
                "http": None,
                "https": None
            }

            response = requests.post(base_url, headers=headers, json=data, proxies=no_proxies, timeout=40)
            response.raise_for_status()

            result_text = response.json()['choices'][0]['message']['content']
            result_text = re.sub(r"```json\s*|```\s*", "", result_text)
            return json.loads(result_text)

        except requests.exceptions.HTTPError as err:
            print(f"\n[API request failed:)] {err.response.status_code} - {err.response.text}")
            break 
        except Exception as e:
            print(f"\nAPI connection error, retrying {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)

    return {"corrections": {}, "new_discoveries": []}
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDataLoader:
    def __init__(self, cfg):
        preprocessor = Preprocessor(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.bert_path)

        self.data = preprocessor.forward()

        self.kernel = WordPair()
        self.config = cfg

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def collate_fn(self, lst):
        if 'dep' in self.config.input_files:
            doc_id, speaker_ids, input_ids, input_masks, input_segments, sentence_length, token2sents, utterance_index, \
                token_index, thread_length, token2speaker, reply_mask, speaker_mask, thread_mask, pieces2words, new2old, \
                triplets, pairs, entity_list, rel_list, polarity_list, thread_idxes, \
                merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, \
                adj_matrixes, utterance_level_reply_adj, utterance_level_speaker_adj, \
                utt_topic_speaker_adj, token_thread_adj, sentence_level_adj = zip(*lst)
        else:
            doc_id, speaker_ids, input_ids, input_masks, input_segments, sentence_length, token2sents, utterance_index, \
                token_index, thread_length, token2speaker, reply_mask, speaker_mask, thread_mask, pieces2words, new2old, \
                triplets, pairs, entity_list, rel_list, polarity_list, thread_idxes, \
                merged_input_ids, merged_input_masks, merged_input_segments, merged_sentence_length, \
                utt_topic_speaker_adj, token_thread_adj, sentence_level_adj = zip(*lst)

        merged_dialog_length = list(map(len, merged_input_ids))
        max_lens = max(map(lambda line: max(map(len, line)), merged_input_ids))
        padding = lambda input_batch: [w + [self.config.pad_id] * (max_lens - len(w)) for line in input_batch for w in
                                       line]
        merged_input_ids = padding(merged_input_ids)
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for line in input_batch for w in line]
        merged_input_masks, merged_input_segments = map(padding, [merged_input_masks, merged_input_segments])

        if 'dep' in self.config.input_files:
            max_lens_mtx = max(map(lambda mtxs: max(map(len, mtxs)), adj_matrixes))
            adj_matrixes = [mtx for sample in adj_matrixes for mtx in sample]
            new_adj_matrixes = np.zeros([len(adj_matrixes), max_lens_mtx, max_lens_mtx], dtype=np.float32)
            for i in range(len(adj_matrixes)):
                new_adj_matrixes[i, :len(adj_matrixes[i]), :len(adj_matrixes[i])] = adj_matrixes[i]

            sent_range = list(map(lambda x: list(accumulate(x)), sentence_length))
            cls_idx = list(map(lambda x: [0] + x[:-1], sent_range))
            sep_idx = list(map(lambda x: [i - 1 for i in x], sent_range))
            doc_length = list(map(lambda x: sum(x), sentence_length))
            sentence_nums = list(map(len, sentence_length))

            global_masks = np.zeros([len(doc_length), max(sentence_nums), max(doc_length), 1], dtype=np.float32)

            for bat_idx, (sent_num, sep_i) in enumerate(zip(sentence_nums, sep_idx)):
                for sent_idx, s_i in zip(range(sent_num), sep_i):
                    global_masks[bat_idx, sent_idx, s_i, :] = 1
            for bat_idx, (sent_num, cls_i, sep_i) in enumerate(zip(sentence_nums, cls_idx, sep_idx)):
                for sent_idx, c_i, s_i in zip(range(sent_num), cls_i, sep_i):
                    global_masks[bat_idx, sent_idx, c_i + 1:s_i, :] = 1

            utterance_len = list(map(len, utterance_level_reply_adj))
            utterance_level_reply_adj_np = np.zeros((len(utterance_len), max(utterance_len), max(utterance_len)),
                                                    dtype=np.float32)
            for i in range(len(utterance_len)):
                utterance_level_reply_adj_np[i, :utterance_len[i], :utterance_len[i]] = utterance_level_reply_adj[i][
                                                                                        :utterance_len[i],
                                                                                        :utterance_len[i]]

            utterance_level_speaker_adj_np = np.zeros((len(utterance_len), max(utterance_len), max(utterance_len)),
                                                      dtype=np.float32)
            for i in range(len(utterance_len)):
                utterance_level_speaker_adj_np[i, :utterance_len[i], :utterance_len[i]] = utterance_level_speaker_adj[
                                                                                              i][:utterance_len[i],
                                                                                          :utterance_len[i]]

            utterance_level_mask = np.zeros([len(utterance_len), max(utterance_len)], dtype=np.float32)
            for i in range(len(utterance_len)):
                utterance_level_mask[i, :utterance_len[i]] = 1

        dialogue_length = list(map(len, input_ids))
        max_lens = max(dialogue_length)
        padding = lambda input_batch: [line + [10] * (max_lens - len(line)) for line in input_batch]
        speaker_ids = padding(speaker_ids)

        max_lens = max(map(lambda line: max(map(len, line)), input_ids))
        input_ids = [w + [self.config.pad_id] * (max_lens - len(w)) for line in input_ids for w in line]
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for line in input_batch for w in line]
        input_masks, input_segments = map(padding, [input_masks, input_segments])

        max_lens = max(map(len, token2sents))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for w in input_batch]
        token2sents, utterance_index, token_index, token2speaker = map(padding,
                                                                       [token2sents, utterance_index, token_index,
                                                                        token2speaker])

        padding_list = lambda input_batch: [list(map(list, w)) + [[0, 0, 0]] * (max(map(len, input_batch)) - len(w)) for
                                            w in input_batch]
        entity_lists, rel_lists, polarity_lists = map(padding_list, [entity_list, rel_list, polarity_list])

        max_tri_num = max(map(len, triplets))
        triplet_masks = [[1] * len(w) + [0] * (max_tri_num - len(w)) for w in triplets]
        triplets = [list(map(list, w)) + [[0] * 7] * (max_tri_num - len(w)) for w in triplets]

        sentence_masks = np.zeros([len(token2sents), max_lens, max_lens], dtype=int)
        for i in range(len(sentence_length)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length[i]]
            masks = block_diag(*masks)
            sentence_masks[i, :len(masks), :len(masks)] = masks
        sentence_masks = sentence_masks.tolist()

        flatten_length = list(map(sum, sentence_length))
        cur_masks = (np.expand_dims(np.arange(max(flatten_length)), 0) < np.expand_dims(flatten_length, 1)).astype(
            np.int64)
        full_masks = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()

        entity_matrix = self.kernel.list2rel_matrix4batch(entity_lists, max_lens)
        rel_matrix = self.kernel.list2rel_matrix4batch(rel_lists, max_lens)
        polarity_matrix = self.kernel.list2rel_matrix4batch(polarity_lists, max_lens)

        new_reply_masks = np.zeros([len(reply_mask), max_lens, max_lens], dtype=np.float32)
        for i in range(len(new_reply_masks)):
            lens = len(reply_mask[i])
            new_reply_masks[i, :lens, :lens] = reply_mask[i]

        new_speaker_masks = np.zeros([len(speaker_mask), max_lens, max_lens], dtype=np.float32)
        for i in range(len(new_speaker_masks)):
            lens = len(speaker_mask[i])
            new_speaker_masks[i, :lens, :lens] = speaker_mask[i]

        new_thread_masks = np.zeros([len(thread_mask), max_lens, max_lens], dtype=np.float32)
        for i in range(len(new_thread_masks)):
            lens = len(thread_mask[i])
            new_thread_masks[i, :lens, :lens] = thread_mask[i]

        max_seq_len = max(len(line) for line in input_ids)

        utt_topic_speaker_adj_padded = np.zeros((len(utt_topic_speaker_adj), max_seq_len, max_seq_len),
                                                dtype=np.float32)
        for i, adj in enumerate(utt_topic_speaker_adj):
            size = min(adj.shape[0], max_seq_len)
            utt_topic_speaker_adj_padded[i, :size, :size] = adj[:size, :size]

        token_thread_adj_padded = np.zeros((len(token_thread_adj), max_seq_len, max_seq_len), dtype=np.float32)
        for i, adj in enumerate(token_thread_adj):
            size = min(adj.shape[0], max_seq_len)
            token_thread_adj_padded[i, :size, :size] = adj[:size, :size]

        sentence_level_adj_padded = np.zeros((len(sentence_level_adj), max_seq_len, max_seq_len), dtype=np.float32)
        for i, adj in enumerate(sentence_level_adj):
            size = min(adj.shape[0], max_seq_len)
            sentence_level_adj_padded[i, :size, :size] = adj[:size, :size]

        res = {
            "doc_id": doc_id, 'speaker_ids': speaker_ids, 'input_ids': input_ids, 'input_masks': input_masks,
            'input_segments': input_segments, 'sentence_length': sentence_length,
            'ent_matrix': entity_matrix, 'rel_matrix': rel_matrix, 'pol_matrix': polarity_matrix,
            'sentence_masks': sentence_masks, 'full_masks': full_masks,
            'triplets': triplets, 'triplet_masks': triplet_masks, 'pairs': pairs,
            'token2sents': token2sents, 'dialogue_length': dialogue_length,
            'utterance_index': utterance_index, 'token_index': token_index,
            'thread_lengths': thread_length, 'token2speakers': token2speaker,
            'reply_masks': new_reply_masks, 'speaker_masks': new_speaker_masks, 'thread_masks': new_thread_masks,
            'pieces2words': pieces2words, 'new2old': new2old, 'thread_idxes': thread_idxes,
            'merged_input_ids': merged_input_ids, 'merged_input_masks': merged_input_masks,
            'merged_input_segments': merged_input_segments,
            'merged_sentence_length': merged_sentence_length, 'merged_dialog_length': merged_dialog_length,
        }

        items = {
            "adj_matrixes": new_adj_matrixes if 'dep' in self.config.input_files else None,
            "global_masks": global_masks if 'dep' in self.config.input_files else None,
            "utterance_level_reply_adj": utterance_level_reply_adj_np if 'dep' in self.config.input_files else None,
            "utterance_level_speaker_adj": utterance_level_speaker_adj_np if 'dep' in self.config.input_files else None,
            "utterance_level_mask": utterance_level_mask if 'dep' in self.config.input_files else None,
            "utt_topic_speaker_graph": utt_topic_speaker_adj_padded,
            "token_thread_graph": token_thread_adj_padded,
            "sentence_level_graph": sentence_level_adj_padded,
        }

        items = {k: v for k, v in items.items() if v is not None}
        res.update(items)

        nocuda = ['sentence_length', 'thread_lengths', 'pairs', 'doc_id', 'pieces2words', 'new2old',
                  'merged_sentence_length', 'merged_dialog_length', 'thread_idxes', 'nonspeaker_token_positions']
        res = {k: v if k in nocuda else torch.tensor(v).to(self.config.device) for k, v in res.items()}
        return res


    def getdata(self):

        def load_data(mode):
            if mode not in self.config.input_files:
                return None
            return DataLoader(MyDataset(self.data[mode]), num_workers=0, worker_init_fn=self.worker_init,
                              shuffle=('train' in mode), batch_size=self.config.batch_size, collate_fn=self.collate_fn)

        modes = self.config.input_files.split()

        loaders = map(load_data, modes)

        line = 'polarity_dict target_dict aspect_dict opinion_dict entity_dict relation_dict'.split()
        for w, z in zip(line, self.data['label_dict']):
            self.config[w] = z

        res = (*loaders, self.config)

        return res


class RelationMetric:
    def __init__(self, config):
        self.clear()
        self.kernel = WordPair()
        self.predict_result = defaultdict(list)
        self.config = config

    def trans2position(self, triplet, new2old, pieces2words):
        res = []
        """
        recover the position of entities in the original sentence

        new2old: transfer position from index with CLS and SEP to index without CLS and SEP
        pieces2words: transfer position from index of wordpiece to index of original words 

        Example:
        list0 (original sentence):"London is the capital of England"
        list1 (tokenized sentence): "Lon ##don is the capital of England"
        list2 (packed sentence): "[CLS] Lon #don is the capital of England [SEP]"
        predicted entity: (1, 2), denotes "Lon #don" in list2

        new2old: list2->list1
          = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, ...}
        pieces2words: list1->list0
          = {'0': 0, '1': 0, '2': 1, '3': 2, '4': 3, ...}

        input  -> entity in list2: "Lon #don" (1, 2)
        middle -> entity in list1: "Lon #don" (0, 1)
        output -> entity in list0: "London"   (0, 0)
        """

        head = lambda x: pieces2words[new2old[x]]
        tail = lambda x: pieces2words[new2old[x]]

        triplet = list(triplet)
        for s0, e0, s1, e1, s2, e2, pol in triplet:
            ns0, ns1, ns2 = head(s0), head(s1), head(s2)
            ne0, ne1, ne2 = tail(e0), tail(e1), tail(e2)
            res.append([ns0, ne0, ns1, ne1, ns2, ne2, pol])
        return res

    def trans2pair(self, pred_pairs, new2old, pieces2words):
        new_pairs = {}
        new_pos = lambda x: pieces2words[new2old[x]]
        for k, line in pred_pairs.items():
            new_line = []
            for s0, e0, s1, e1 in line:
                s0, e0, s1, e1 = new_pos(s0), new_pos(e0), new_pos(s1), new_pos(e1)
                new_line.append([s0, e0, s1, e1])
            new_pairs[k] = new_line
        return new_pairs

    def filter_entity(self, ent_list, new2old, pieces2words):
        res = []

        # If the entity is a sub-string of another entity, remove it
        # ent_list = sorted(ent_list, key=lambda x: (x[0], -x[1]))
        # ent_list = [w for i, w in enumerate(ent_list) if i == 0 or w[0] != ent_list[i-1][0]]

        for s, e, pol in ent_list:
            ns, ne = pieces2words[new2old[s]], pieces2words[new2old[e]]
            res.append([ns, ne, pol])
        return res

    def add_instance(self, data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix, use_llm=False):
        import torch
        import numpy as np
        token2sents = data['token2sents'].tolist()
        new2old = data['new2old']
        pieces2words = data['pieces2words']
        doc_id = data['doc_id']

        ori_ent_logits = pred_ent_matrix.clone()
        ori_rel_logits = pred_rel_matrix.clone()
        ori_pol_logits = pred_pol_matrix.clone()

        if use_llm:
            if not hasattr(self, 'tokenizer_llm'):
                from transformers import AutoTokenizer
                self.tokenizer_llm = AutoTokenizer.from_pretrained(self.config.bert_path)

            ent_probs = torch.softmax(ori_ent_logits, dim=-1)
            rel_probs = torch.softmax(ori_rel_logits, dim=-1)
            pol_probs = torch.softmax(ori_pol_logits, dim=-1)

            inv_ent_dic = {0: "背景", 1: "Target", 2: "Aspect", 3: "Opinion"}

            for b in range(len(pred_ent_matrix)):
                seq_len = ent_probs.shape[1]
                llm_payload = {"matrix_1_entity_choices": [], "matrix_2_relation_choices": []}
                task_mapping = {}
                choice_idx = 0

                merged_ids = data['merged_input_ids'][b].cpu().tolist()
                tokens = self.tokenizer_llm.convert_ids_to_tokens(merged_ids)

                def get_word(x, y):
                    if x < 0 or y >= len(tokens) or x > y: return "非法边界"
                    span_tokens = tokens[x:y + 1]
                    text = "".join(span_tokens).replace("##", "")
                    clean_text = text.replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "").strip()
                    # 过滤单字符标点
                    if len(clean_text) == 1 and clean_text in [",", ".", "!", "?", "，", "。", "！", "？", "[", "]", "(",
                                                               ")"]:
                        return "特殊符号"
                    return clean_text if clean_text else "特殊符号"

                top3_ent_probs, top3_ent_indices = torch.topk(ent_probs[b], 3, dim=-1)
                candidate_spans_int = []

                for x in range(seq_len):
                    for y in range(x, seq_len):
                        if data['sentence_masks'][b][x][y] == 0: continue

                        word_text = get_word(x, y)
                        if "非法边界" in word_text or "特殊符号" in word_text or len(word_text) == 0:
                            continue

                        p1, c1 = top3_ent_probs[x, y, 0].item(), top3_ent_indices[x, y, 0].item()
                        p2, c2 = top3_ent_probs[x, y, 1].item(), top3_ent_indices[x, y, 1].item()
                        p3, c3 = top3_ent_probs[x, y, 2].item(), top3_ent_indices[x, y, 2].item()

                        if (p1 - p3) < 0.6 and max(c1, c2, c3) != 0:
                            task_id = f"ent_choice_{choice_idx}"
                            task_mapping[task_id] = {
                                "matrix": "ent", "coord": (x, y),
                                "opt_1": c1, "opt_2": c2, "opt_3": c3,  # 记录三个选项
                                "word": word_text,
                                "opt_1_name": inv_ent_dic[c1], "opt_2_name": inv_ent_dic[c2],
                                "opt_3_name": inv_ent_dic[c3]
                            }
                            llm_payload["matrix_1_entity_choices"].append({
                                "id": task_id, "issue_type": "类别犹豫", "word": word_text,
                                "option_1": {"type": inv_ent_dic[c1], "prob": round(p1, 3)},
                                "option_2": {"type": inv_ent_dic[c2], "prob": round(p2, 3)},
                                "option_3": {"type": inv_ent_dic[c3], "prob": round(p3, 3)}
                            })
                            choice_idx += 1

                        max_ent_prob = max(ent_probs[b, x, y, 1].item(), ent_probs[b, x, y, 2].item(),
                                           ent_probs[b, x, y, 3].item())
                        if max_ent_prob > 0.2:
                            best_class = 1 if max_ent_prob == ent_probs[b, x, y, 1].item() else (
                                2 if max_ent_prob == ent_probs[b, x, y, 2].item() else 3)
                            candidate_spans_int.append(
                                {"coord": (x, y), "class": best_class, "prob": max_ent_prob, "word": word_text})


                num_tasks = len(llm_payload["matrix_1_entity_choices"])

                if num_tasks > 0:
                    max_tasks = 8
                    if num_tasks > max_tasks:
                        llm_payload["matrix_1_entity_choices"] = llm_payload["matrix_1_entity_choices"][:max_tasks]

                    dialogue_words = [
                        tok.replace("##", "") for tok in tokens 
                        if tok not in ["[PAD]", "[CLS]", "[SEP]"]
                    ]

                    if not dialogue_words:
                        continue

                    dialogue_text = "".join(dialogue_words)
                    llm_response = request_llm_judge(llm_payload, dialogue_text)

                    # Update logits based on LLM calibration (Eq. 35)
                    for task_id, decision in llm_response.get("corrections", {}).items():
                        chosen = decision.get("choice", "none") if isinstance(decision, dict) else decision
                        if task_id not in task_mapping or chosen == "none":
                            continue

                        meta = task_mapping[task_id]
                        if meta["matrix"] == "ent":
                            if chosen == "option_1":
                                correct_class = meta["opt_1"]
                            elif chosen == "option_2":
                                correct_class = meta["opt_2"]
                            elif chosen == "option_3":
                                correct_class = meta.get("opt_3", 0)
                            else:
                                continue

                            # Directional boost: lambda = 1.2
                            ori_ent_logits[b, meta["coord"][0], meta["coord"][1], correct_class] += 1.2


        final_ent_matrices = (ori_ent_logits.argmax(-1) * data['sentence_masks']).cpu().numpy()
        final_rel_matrices = (ori_rel_logits.argmax(-1) * data['full_masks']).cpu().numpy()
        final_pol_matrices = (ori_pol_logits.argmax(-1) * data['full_masks']).cpu().numpy()

        for i in range(len(final_ent_matrices)):
            ent_matrix, rel_matrix, pol_matrix = final_ent_matrices[i], final_rel_matrices[i], final_pol_matrices[i]

            pred_triplet, pred_pairs = self.kernel.get_triplets(ent_matrix, rel_matrix, pol_matrix, token2sents[i])
            pred_ents = self.kernel.rel_matrix2list(ent_matrix)

            pred_ents = self.filter_entity(pred_ents, new2old[i], pieces2words[i])
            pred_pairs = self.trans2pair(pred_pairs, new2old[i], pieces2words[i])
            pred_triplet = self.trans2position(pred_triplet, new2old[i], pieces2words[i])

            # 存入结果，等待评测
            self.predict_result[doc_id[i]].append(pred_ents)
            self.predict_result[doc_id[i]].append(pred_pairs)
            self.predict_result[doc_id[i]].append(pred_triplet)

    def clear(self):
        self.predict_result = defaultdict(list)

    def save2file(self, gold_file, pred_file):
        # pol_dict = {"O": 0, "pos": 1, "neg": 2, "other": 3}
        pol_dict = self.config.polarity_dict
        reverse_pol_dict = {v: k for k, v in pol_dict.items()}
        reverse_ent_dict = {v: k for k, v in self.config.entity_dict.items()}

        gold_file = open(gold_file, 'r', encoding='utf-8')

        data = json.load(gold_file)

        res = []
        for line in data:
            doc_id, sentence = line['doc_id'], line['sentences']
            if doc_id not in self.predict_result:
                continue
            doc = ' '.join(sentence).split()
            new_triples = []

            prediction = self.predict_result[doc_id]
            entities = defaultdict(list)
            for head, tail, tp in prediction[0]:
                tp = reverse_ent_dict[tp]
                head, tail = head, tail + 1
                tp_dict = {'ENT-T': 'targets', 'ENT-A': 'aspects', 'ENT-O': 'opinions'}
                entities[tp_dict[tp]].append([head, tail])

            pairs = defaultdict(list)
            for key in ['ta', 'to', 'ao']:
                for s0, e0, s1, e1 in prediction[1][key]:
                    e0, e1 = e0 + 1, e1 + 1
                    pairs[key].append([s0, e0, s1, e1])

            new_triples = []
            for s0, e0, s1, e1, s2, e2, pol in prediction[2]:
                pol = reverse_pol_dict[pol]
                e0, e1, e2 = e0 + 1, e1 + 1, e2 + 1
                new_triples.append(
                    [s0, e0, s1, e1, s2, e2, pol, ' '.join(doc[s0:e0]), ' '.join(doc[s1:e1]), ' '.join(doc[s2:e2])])

            res.append({'doc_id': doc_id, 'triplets': new_triples, \
                        'targets': entities['targets'], 'aspects': entities['aspects'],
                        'opinions': entities['opinions'], \
                        'ta': pairs['ta'], 'to': pairs['to'], 'ao': pairs['ao']})
        logger.info('Save prediction results to {}'.format(pred_file))
        json.dump(res, open(pred_file, 'w', encoding='utf-8'), ensure_ascii=False)

    def compute(self, name='valid', action='eval', msg=""):
        # action: pred, make prediction, save to file
        # action: eval, make prediction, save to file and evaluate

        gold_file = os.path.join(self.config.json_path, '{}.json'.format(name))
        if self.config.testset_name is not None:
            gold_file = os.path.join(self.config.json_path, '{}'.format(self.config.testset_name))
        args = AttrDict({
            'pred_file': os.path.join(self.config.target_dir, 'pred_{}_{}_{}.json'.format(self.config.lang, name, msg)),
            'gold_file': gold_file
        })

        self.save2file(args.gold_file, args.pred_file)
        if action == 'pred':
            return
        micro, iden, res = Run_eval(args).forward()
        self.clear()
        return micro[2], iden[2], res


def log_unhandled_exceptions(exctype, value, traceback):
    logger.exception(f"Uncaught exception: {value}", exc_info=(exctype, value, traceback))
    traceback.print_tb(traceback)

