#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证流式实现是否成功

运行: python scripts/verify_streaming.py

检查项:
1. 数据时间戳按全局顺序（流式输入）
2. streaming_state 在 batch 间正确持久化
3. 同一用户在连续 batch 中会复用上一 batch 的隐状态
4. 不同用户/不同 batch 间状态隔离
"""

import sys
import torch

# Add project root
sys.path.insert(0, ".")


def verify_data_timestamp_order():
    """验证训练数据按时间戳全局升序（流式输入的核心）"""
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    config_dict = {
        "model": "GDN",
        "dataset": "ml-100k",
        "embedding_size": 32,
        "hidden_size": 64,
        "streaming_mode": True,
        "train_neg_sample_args": None,
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "none",
            "order": "TO",
            "mode": "full",
        },
        "shuffle": False,
    }
    config = Config(model="GDN", dataset="ml-100k", config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    time_field = config["TIME_FIELD"]
    timestamps = []

    for batch in train_data:
        if time_field in batch:
            ts = batch[time_field]
            if isinstance(ts, torch.Tensor):
                timestamps.extend(ts.cpu().tolist())
            else:
                timestamps.extend(ts)

    assert len(timestamps) > 0, "未找到时间戳字段"
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"时间戳非升序: 位置 {i} 的 {timestamps[i]} < 位置 {i-1} 的 {timestamps[i-1]}"
        )

    print(f"[PASS] 数据时间戳顺序: 共 {len(timestamps)} 条，全局升序")


def verify_streaming_state_persistence():
    """验证 _streaming_state 在 batch 间正确持久化"""
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.sequential_recommender.gdn import GDN

    config_dict = {
        "model": "GDN",
        "dataset": "ml-100k",
        "embedding_size": 32,
        "streaming_mode": True,
        "train_neg_sample_args": None,
    }
    config = Config(model="GDN", dataset="ml-100k", config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    model = GDN(config, train_data._dataset)
    model.eval()

    # 清空状态
    model.reset_streaming_state()
    assert len(model._streaming_state) == 0, "初始状态应为空"

    # 模拟重叠序列：batch2 是 batch1 的扩展，只应增量更新
    device = next(model.parameters()).device
    batch1_user = torch.tensor([1], device=device)
    batch1_seq = torch.tensor([[10, 20, 30]], device=device)  # [A,B,C]
    batch1_len = torch.tensor([3], device=device)

    with torch.no_grad():
        out1 = model.forward_with_streaming(batch1_seq, batch1_len, batch1_user)

    assert 1 in model._streaming_state, "Batch1 后 user 1 的状态应被持久化"
    # 多层级时 state[0] 为 (S_layer0, S_layer1, ...)
    S_after_batch1 = model._streaming_state[1][0][0].clone()

    # Batch2: [A,B,C,D,E] 扩展序列，只应处理 D,E（增量）
    batch2_user = torch.tensor([1], device=device)
    batch2_seq = torch.tensor([[10, 20, 30, 40, 50]], device=device)
    batch2_len = torch.tensor([5], device=device)

    S_before_batch2 = model._streaming_state[1][0][0].clone()

    with torch.no_grad():
        out2 = model.forward_with_streaming(batch2_seq, batch2_len, batch2_user)

    # Batch2 初始 S 应等于 Batch1 结束的 S
    diff = (S_before_batch2 - S_after_batch1).abs().max()
    assert diff.item() < 1e-5, f"Batch2 应复用 Batch1 的 S，差异={diff.item()}"

    # Batch2 后 S 应更新（处理了 40,50）
    S_after_batch2 = model._streaming_state[1][0][0].clone()
    assert (S_after_batch2 != S_after_batch1).any(), "Batch2 增量更新后 S 应变化"

    print("[PASS] 流式状态持久化: 同一用户跨 batch 正确复用并更新状态")


def verify_state_isolation():
    """验证不同用户状态隔离"""
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.sequential_recommender.gdn import GDN

    config_dict = {
        "model": "GDN",
        "dataset": "ml-100k",
        "embedding_size": 32,
        "streaming_mode": True,
        "train_neg_sample_args": None,
    }
    config = Config(model="GDN", dataset="ml-100k", config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    model = GDN(config, train_data._dataset)
    model.eval()
    model.reset_streaming_state()

    device = next(model.parameters()).device

    # Batch: user 1 和 user 2
    users = torch.tensor([1, 2], device=device)
    seq = torch.tensor([[10, 20], [30, 40]], device=device)
    seq_len = torch.tensor([2, 2], device=device)

    with torch.no_grad():
        model.forward_with_streaming(seq, seq_len, users)

    assert 1 in model._streaming_state and 2 in model._streaming_state
    # 两个用户的 S 矩阵应不同（取首层）
    diff = (model._streaming_state[1][0][0] - model._streaming_state[2][0][0]).abs().max()
    assert diff.item() > 1e-5, "不同用户应有不同状态"

    print("[PASS] 流式状态隔离: 不同用户状态独立")


def verify_reset_clears_state():
    """验证 reset_streaming_state 正确清空"""
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.sequential_recommender.gdn import GDN

    config_dict = {
        "model": "GDN",
        "dataset": "ml-100k",
        "embedding_size": 32,
        "streaming_mode": True,
        "train_neg_sample_args": None,
    }
    config = Config(model="GDN", dataset="ml-100k", config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    model = GDN(config, train_data._dataset)
    model.eval()

    device = next(model.parameters()).device
    users = torch.tensor([1], device=device)
    seq = torch.tensor([[10, 20]], device=device)
    seq_len = torch.tensor([2], device=device)

    with torch.no_grad():
        model.forward_with_streaming(seq, seq_len, users)

    assert len(model._streaming_state) == 1
    model.reset_streaming_state()
    assert len(model._streaming_state) == 0, "reset 后状态应被清空"

    print("[PASS] reset_streaming_state: 正确清空状态")


def main():
    print("=" * 50)
    print("流式实现验证")
    print("=" * 50)

    try:
        verify_data_timestamp_order()
        verify_streaming_state_persistence()
        verify_state_isolation()
        verify_reset_clears_state()
        print()
        print("[OK] 所有流式验证通过")
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
