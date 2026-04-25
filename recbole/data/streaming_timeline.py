# -*- coding: utf-8 -*-
# Streaming Timeline: 全局时间轴 + Test-Then-Train + 无限长程记忆
#
# 按真实世界时间戳排序，逐条喂数据。
# 对每条交互：若为 test 点则先预测(算 NDCG/Recall)，再算 loss 更新，最后更新用户 S。
# 用户 S 永不重置，伴随生命周期演化。
#
# 若配置 streaming_pretrain_dataset，则用预训练集历史初始化 user_history，
# 否则 T2T 流式阶段仅含 20% 数据，序列过短导致 recall 异常偏低。
#
# 词表对齐：RecBole 的 pd.factorize 按首次出现顺序分配 ID，pretrain 与 t2t 的 inter 顺序不同
# 会导致同一 token 得到不同内部 ID，预训练 Embedding 与 T2T 数据错位。必须用预训练词表覆盖 T2T。

import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from logging import getLogger

from recbole.data.interaction import Interaction


def _norm_tok(t):
    """将 RecBole atomic 里混用的 int/float/numpy.str/字符串 统一成可比较的 id 串（Yelp 必用）。"""
    if t is None:
        return ""
    if hasattr(t, "item") and not isinstance(t, (str, bytes)):
        try:
            t = t.item()
        except Exception:
            pass
    if isinstance(t, (bool, np.bool_)):
        return str(t).strip()
    if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
        return str(int(t))
    if isinstance(t, (float, np.floating)):
        f = float(t)
        if np.isnan(f) or np.isinf(f):
            return ""
        if f == int(f):
            return str(int(f))
        return str(t).strip()
    s = str(t).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    try:
        f = float(s)
        if not (np.isnan(f) or np.isinf(f)) and f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


def _field_id2token_list(dataset, field):
    """RecBole 的 field2id_token[field] 可能是 list 或 np.ndarray，不能用 `x or []`（ndarray 不能当 bool）。"""
    v = None
    if hasattr(dataset, "field2id_token") and dataset.field2id_token is not None:
        v = dataset.field2id_token.get(field)
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


def _canon_token_to_id_map(pretrain_ds, field):
    """构建 规范化 token 串 -> 预训练内部 id。扫 field2token_id 与 field2id_token 两套来源，键类型全吃齐。

    仅 dict.get(原始key) 在 Yelp 上会大量 OOV→0，因键可能是 numpy 标量、'123'/'123.0' 等。
    """
    tid = pretrain_ds.field2token_id.get(field) if hasattr(pretrain_ds, "field2token_id") else None
    if tid is None:
        tid = {}
    id2t = _field_id2token_list(pretrain_ds, field)
    out = {}
    for k, v in tid.items():
        ck = _norm_tok(k)
        if ck and ck not in out:
            out[ck] = int(v)
    for i, tok in enumerate(id2t):
        if i == 0:
            continue
        ck = _norm_tok(tok)
        if not ck or ck in out:
            continue
        v = tid.get(tok)
        if v is None:
            v = tid.get(str(tok))
        if v is None:
            v = i
        out[ck] = int(v)
    return out



def _align_t2t_vocab_to_pretrain(t2t_dataset, pretrain_dataset, config):
    """将 T2T 数据集的词表与预训练对齐，并重映射所有 ID 张量。
    RecBole 用 pd.factorize 按首次出现顺序分配 ID，pretrain/t2t 的 inter 顺序不同导致 ID 错位。
    """
    logger = getLogger()
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    item_id_list_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]

    def _build_id_map(t2t_ds, pretrain_ds, field):
        """t2t_id -> pretrain_id，0(PAD) 保持 0"""
        t2t_id2token = t2t_ds.field2id_token.get(field)
        if t2t_id2token is None:
            return None
        canon = _canon_token_to_id_map(pretrain_ds, field)
        if not canon:
            return None
        id_map = np.zeros(len(t2t_id2token), dtype=np.int64)
        n_miss = 0
        for old_id in range(len(t2t_id2token)):
            if old_id == 0:
                id_map[0] = 0
                continue
            token = t2t_id2token[old_id]
            ck = _norm_tok(token)
            pid = int(canon.get(ck, 0)) if ck else 0
            id_map[old_id] = pid
            if old_id > 0 and pid == 0 and ck:
                n_miss += 1
        if n_miss > 0:
            getLogger().debug(
                "[StreamingTimeline] %s: id_map 中仍 OOV token 数 %d / %d（仅 debug）",
                field,
                n_miss,
                max(len(t2t_id2token) - 1, 1),
            )
        if len(id_map) > 1 and int(np.sum(id_map[1:])) == 0:
            sm = []
            for j in range(1, min(6, len(t2t_id2token))):
                tok = t2t_id2token[j]
                sm.append((j, repr(tok), _norm_tok(tok)))
            getLogger().error(
                "[StreamingTimeline] %s 的 id_map 除 0 外全 OOV；t2t 样例 (internal_id, raw, norm)=%s。"
                "若与 pretrain 同源仍如此，请核对两子集 .inter 中该列是否一致。",
                field,
                sm,
            )
        return id_map

    uid_map = _build_id_map(t2t_dataset, pretrain_dataset, uid_field)
    iid_map = _build_id_map(t2t_dataset, pretrain_dataset, iid_field)
    if uid_map is None or iid_map is None:
        logger.warning("[StreamingTimeline] 词表对齐失败：缺少 field2token_id")
        return
    if len(uid_map) > 1:
        logger.info(
            "[StreamingTimeline] uid_map 非零项 %d / %d（t2t user 词表槽，不含 PAD=0）",
            int(np.sum(uid_map[1:] > 0)),
            len(uid_map) - 1,
        )
    if len(iid_map) > 1:
        logger.info(
            "[StreamingTimeline] iid_map 非零项 %d / %d",
            int(np.sum(iid_map[1:] > 0)),
            len(iid_map) - 1,
        )

    def _remap_and_assign(container, key, m):
        if key not in container or m is None:
            return
        arr = container[key]
        arr_np = arr.cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
        new_vals = np.take(m, np.clip(arr_np.astype(np.int64), 0, len(m) - 1))
        new_vals = np.where(arr_np < len(m), new_vals, 0)
        if torch.is_tensor(arr):
            container[key] = torch.tensor(new_vals, dtype=arr.dtype, device=arr.device)
        else:
            container[key] = new_vals

    # 重映射 inter_feat
    inter = t2t_dataset.inter_feat
    inter_dict = inter.interaction if hasattr(inter, "interaction") else inter
    _remap_and_assign(inter_dict, uid_field, uid_map)
    _remap_and_assign(inter_dict, iid_field, iid_map)
    if item_id_list_field in inter_dict:
        arr = inter_dict[item_id_list_field]
        arr_np = arr.cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
        new_arr = np.take(iid_map, np.clip(arr_np.astype(np.int64), 0, len(iid_map) - 1))
        new_arr = np.where(arr_np < len(iid_map), new_arr, 0)
        inter_dict[item_id_list_field] = (
            torch.tensor(new_arr, dtype=arr.dtype, device=arr.device)
            if torch.is_tensor(arr)
            else new_arr
        )

    # 重映射 _raw_inter_for_timeline（用于 timeline 构建）
    if hasattr(t2t_dataset, "_raw_inter_for_timeline") and t2t_dataset._raw_inter_for_timeline is not None:
        raw = t2t_dataset._raw_inter_for_timeline.interaction
        _remap_and_assign(raw, uid_field, uid_map)
        _remap_and_assign(raw, iid_field, iid_map)

    # 覆盖词表
    t2t_dataset.field2id_token[uid_field] = pretrain_dataset.field2id_token[uid_field]
    t2t_dataset.field2token_id[uid_field] = pretrain_dataset.field2token_id[uid_field]
    t2t_dataset.field2id_token[iid_field] = pretrain_dataset.field2id_token[iid_field]
    t2t_dataset.field2token_id[iid_field] = pretrain_dataset.field2token_id[iid_field]

    # item_id_list 与 item_id 共享词表
    if item_id_list_field in t2t_dataset.field2id_token:
        t2t_dataset.field2id_token[item_id_list_field] = pretrain_dataset.field2id_token[iid_field]
        t2t_dataset.field2token_id[item_id_list_field] = pretrain_dataset.field2token_id[iid_field]

    # 诊断：若对齐后 T2T 的 user 列几乎为常数，流式 T2T 会全 0 或越界
    if hasattr(t2t_dataset, "_raw_inter_for_timeline") and t2t_dataset._raw_inter_for_timeline is not None:
        rawx = t2t_dataset._raw_inter_for_timeline.interaction
        if uid_field in rawx:
            uu = rawx[uid_field]
            arr = uu.cpu().numpy() if torch.is_tensor(uu) else np.asarray(uu, dtype=np.int64)
            flat = arr.flatten()
            nuniq = len(np.unique(flat))
            n0 = int(np.sum(flat == 0))
            if nuniq <= 1 and len(flat) > 50:
                logger.error(
                    "[StreamingTimeline] 对齐后 T2T user 列仍几乎常数 (unique=%d, n=%d, n_uid==0=%d)。"
                    "多为 t2t 与 pretrain 的 token 键不一致，已全部映成 0。请检查两子集 .inter 的 user 列与 RecBole token 类型。",
                    nuniq,
                    len(flat),
                    n0,
                )
            elif n0 > len(flat) * 0.4:
                logger.warning(
                    "[StreamingTimeline] 对齐后 user 列 label==0 占比 %.1f%%（n=%d），若异常请检查 OOV。",
                    100.0 * n0 / max(len(flat), 1),
                    len(flat),
                )

    logger.info(
        "[StreamingTimeline] 词表已对齐到预训练集：field2token_id/field2id_token 已覆盖，"
        "inter_feat 与 _raw_inter_for_timeline 已重映射"
    )


class StreamingTimelineBuilder:
    """构建全局时间轴，按时间戳排序的 (user_id, item_id, is_test) 序列。"""

    def __init__(self, inter_feat, uid_field, iid_field, time_field, test_ratio=0.1, rating_field=None):
        self.uid_field = uid_field
        self.iid_field = iid_field
        self.time_field = time_field
        self.test_ratio = test_ratio

        data = inter_feat.interaction if hasattr(inter_feat, 'interaction') else inter_feat
        uids = data[self.uid_field].numpy() if torch.is_tensor(data[self.uid_field]) else data[self.uid_field].values
        iids = data[self.iid_field].numpy() if torch.is_tensor(data[self.iid_field]) else data[self.iid_field].values
        times = data[self.time_field].numpy() if torch.is_tensor(data[self.time_field]) else data[self.time_field].values

        # 过滤 rating=0 的占位行（prepare 脚本注入的 vocab 对齐 dummy）
        if rating_field is not None and rating_field in data:
            ratings = data[rating_field].numpy() if torch.is_tensor(data[rating_field]) else data[rating_field].values
            mask = ratings != 0
            uids, iids, times = uids[mask], iids[mask], times[mask]
            n_dropped = int((~mask).sum())
            if n_dropped:
                getLogger().info(f"[StreamingTimeline] 过滤占位行 (rating=0): {n_dropped} 条")

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


def _load_pretrain_user_histories(config, dataset):
    """加载预训练集用户历史，用于 T2T 流式开始时初始化 user_history。
    返回 {internal_uid: [iid1, iid2, ...]}，按时间排序。
    """
    pretrain_name = config.final_config_dict.get("streaming_pretrain_dataset", None)
    if not pretrain_name:
        return {}
    data_path = config["data_path"]
    base_path = os.path.dirname(data_path)
    inter_path = os.path.join(base_path, pretrain_name, f"{pretrain_name}.inter")
    if not os.path.isfile(inter_path):
        getLogger().warning(
            f"[StreamingTimeline] streaming_pretrain_dataset={pretrain_name} 但文件不存在: {inter_path}，跳过预训练历史"
        )
        return {}
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    time_field = config["TIME_FIELD"]
    uid2id_raw = dataset.field2token_id.get(uid_field, {})
    iid2id_raw = dataset.field2token_id.get(iid_field, {})
    if not uid2id_raw or not iid2id_raw:
        return {}
    # 规范化键：RecBole 对整数 ID（如 ML-10M）用 pd.factorize 返回 numpy.int64 键，
    # 而 str(row[col]) 得到 Python str，两者 `in` 判断会全部失配 → 0 users。
    uid2id = {_norm_tok(k): v for k, v in uid2id_raw.items()}
    iid2id = {_norm_tok(k): v for k, v in iid2id_raw.items()}
    encoding = config.final_config_dict.get("encoding", "utf-8")
    df = pd.read_csv(inter_path, sep="\t", encoding=encoding)
    cols = {c.split(":")[0]: c for c in df.columns if ":" in c}
    uid_col = cols.get(uid_field) or uid_field
    if uid_col not in df.columns:
        uid_col = next((c for c in df.columns if c.startswith(uid_field)), df.columns[0])
    iid_col = cols.get(iid_field) or iid_field
    if iid_col not in df.columns:
        iid_col = next((c for c in df.columns if c.startswith(iid_field)), df.columns[1])
    time_col = cols.get(time_field) or time_field
    if time_col not in df.columns:
        time_col = next((c for c in df.columns if "time" in c.lower()), df.columns[-1])
    user_items = defaultdict(list)
    for _, row in df.iterrows():
        utok = _norm_tok(row[uid_col])
        itok = _norm_tok(row[iid_col])
        t = row[time_col]
        if utok not in uid2id or itok not in iid2id:
            continue
        uid, iid = uid2id[utok], iid2id[itok]
        user_items[uid].append((t, iid))
    for uid in user_items:
        user_items[uid].sort(key=lambda x: x[0])
        user_items[uid] = [iid for _, iid in user_items[uid]]
    n_users = len(user_items)
    n_inters = sum(len(v) for v in user_items.values())
    getLogger().info(
        f"[StreamingTimeline] 预训练历史: {pretrain_name}, {n_users} users, {n_inters} interactions"
    )
    return dict(user_items)


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
        initial_user_history=None,
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
        self._initial_user_history = initial_user_history or {}

    def __iter__(self):
        self.user_history.clear()
        for uid, hist in self._initial_user_history.items():
            self.user_history[uid] = list(hist)
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

    rating_field = config.final_config_dict.get("RATING_FIELD", None)
    builder = StreamingTimelineBuilder(
        raw_inter, uid_field, iid_field, time_field,
        test_ratio=test_ratio, rating_field=rating_field,
    )
    initial_user_history = _load_pretrain_user_histories(config, dataset)
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
        initial_user_history=initial_user_history,
    )
