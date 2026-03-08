# -*- coding: utf-8 -*-
# Streaming Timeline: 全局时间轴 + Test-Then-Train + 无限长程记忆
#
# 按真实世界时间戳排序，逐条喂数据。
# 对每条交互：若为 test 点则先预测(算 NDCG/Recall)，再算 loss 更新，最后更新用户 S。
# 用户 S 永不重置，伴随生命周期演化。

import numpy as np
import torch
from collections import defaultdict
from logging import getLogger

from recbole.data.interaction import Interaction


class StreamingTimelineBuilder:
    """构建全局时间轴，按时间戳排序的 (user_id, item_id, is_test) 序列。"""

    def __init__(self, inter_feat, uid_field, iid_field, time_field, test_ratio=0.1):
        self.uid_field = uid_field
        self.iid_field = iid_field
        self.time_field = time_field
        self.test_ratio = test_ratio

        data = inter_feat.interaction if hasattr(inter_feat, 'interaction') else inter_feat
        uids = data[self.uid_field].numpy() if torch.is_tensor(data[self.uid_field]) else data[self.uid_field].values
        iids = data[self.iid_field].numpy() if torch.is_tensor(data[self.iid_field]) else data[self.iid_field].values
        times = data[self.time_field].numpy() if torch.is_tensor(data[self.time_field]) else data[self.time_field].values

        user_interactions = defaultdict(list)
        for uid, iid, t in zip(uids, iids, times):
            user_interactions[uid].append((iid, t))
        for uid in user_interactions:
            user_interactions[uid].sort(key=lambda x: x[1])

        self.user_test_indices = {}
        for uid, inters in user_interactions.items():
            n = len(inters)
            n_test = max(1, int(n * test_ratio))
            self.user_test_indices[uid] = set(range(n - n_test, n))

        timeline = []
        for uid, inters in user_interactions.items():
            for local_idx, (iid, t) in enumerate(inters):
                timeline.append((uid, iid, t, local_idx))
        timeline.sort(key=lambda x: (x[2], x[0]))

        self.timeline = timeline
        self.logger = getLogger()
        self.logger.info(
            f"[StreamingTimeline] {len(timeline)} interactions, "
            f"{len(user_interactions)} users, test_ratio={test_ratio}"
        )

    def __len__(self):
        return len(self.timeline)

    def get_timeline(self):
        for uid, iid, t, local_idx in self.timeline:
            is_test = local_idx in self.user_test_indices[uid]
            yield uid, iid, is_test


class StreamingTimelineDataLoader:
    """按全局时间轴迭代，每 batch 返回 Interaction。"""

    def __init__(
        self,
        timeline_builder,
        dataset,
        batch_size=256,
        max_item_list_len=50,
        uid_field="user_id",
        iid_field="item_id",
        item_id_list_field="item_id_list",
        item_list_length_field="item_list_length",
        device=None,
    ):
        self.builder = timeline_builder
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_item_list_len = max_item_list_len
        self.uid_field = uid_field
        self.iid_field = iid_field
        self.item_id_list_field = item_id_list_field
        self.item_list_length_field = item_list_length_field
        self.device = device
        self.user_history = defaultdict(list)
        self._dataset = dataset  # RecBole compatibility

    def __iter__(self):
        self.user_history.clear()
        batch_uids, batch_seqs, batch_lens, batch_pos, batch_is_test = [], [], [], [], []
        for uid, iid, is_test in self.builder.get_timeline():
            history = self.user_history[uid]
            seq = history[-self.max_item_list_len:] if history else []
            seq_tensor = torch.tensor(seq, dtype=torch.long) if seq else torch.zeros(0, dtype=torch.long)
            seq_len = len(seq)

            batch_uids.append(uid)
            batch_seqs.append(seq_tensor)
            batch_lens.append(seq_len)
            batch_pos.append(iid)
            batch_is_test.append(is_test)

            if len(batch_uids) >= self.batch_size:
                yield self._make_batch(batch_uids, batch_seqs, batch_lens, batch_pos, batch_is_test)
                for u, p in zip(batch_uids, batch_pos):
                    self.user_history[u].append(p)
                batch_uids, batch_seqs, batch_lens, batch_pos, batch_is_test = [], [], [], [], []

        if batch_uids:
            yield self._make_batch(batch_uids, batch_seqs, batch_lens, batch_pos, batch_is_test)
            for u, p in zip(batch_uids, batch_pos):
                self.user_history[u].append(p)

    def _make_batch(self, uids, seqs, lens, pos_items, is_tests):
        max_len = max(lens) if lens else 0
        min_seq_len = 3  # conv kernel needs at least 3
        max_len = max(max_len, min_seq_len)
        pad_seqs = []
        for seq, length in zip(seqs, lens):
            if length < max_len:
                pad = torch.zeros(max_len - length, dtype=torch.long)
                padded = torch.cat([seq, pad])
            else:
                padded = seq
            pad_seqs.append(padded)
        item_seq = torch.stack(pad_seqs) if pad_seqs else torch.zeros(0, 0, dtype=torch.long)
        item_seq_len = torch.tensor(lens, dtype=torch.long)
        user_ids = torch.tensor(uids, dtype=torch.long)
        pos_items_t = torch.tensor(pos_items, dtype=torch.long)
        is_test_t = torch.tensor(is_tests, dtype=torch.bool)

        inter_dict = {
            self.uid_field: user_ids,
            self.item_id_list_field: item_seq,
            self.item_list_length_field: item_seq_len,
            self.iid_field: pos_items_t,
            "is_test": is_test_t,
        }
        return Interaction(inter_dict)

    def __len__(self):
        return (len(self.builder) + self.batch_size - 1) // self.batch_size


def create_streaming_timeline_dataloader(config, dataset):
    """创建流式时间轴 DataLoader。需要 dataset 为 StreamingSequentialDataset 且已设置 _raw_inter_for_timeline。"""
    if not hasattr(dataset, "_raw_inter_for_timeline"):
        raise ValueError(
            "create_streaming_timeline_dataloader 需要 StreamingSequentialDataset (streaming_t2t=True)"
        )
    raw_inter = dataset._raw_inter_for_timeline
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    time_field = config["TIME_FIELD"]
    item_id_list_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
    test_ratio = config.final_config_dict.get("streaming_test_ratio", 0.1)

    builder = StreamingTimelineBuilder(
        raw_inter, uid_field, iid_field, time_field, test_ratio=test_ratio
    )
    return StreamingTimelineDataLoader(
        builder,
        dataset,
        batch_size=config["train_batch_size"],
        max_item_list_len=config["MAX_ITEM_LIST_LENGTH"],
        uid_field=uid_field,
        iid_field=iid_field,
        item_id_list_field=item_id_list_field,
        item_list_length_field=item_list_length_field,
        device=config["device"],
    )
