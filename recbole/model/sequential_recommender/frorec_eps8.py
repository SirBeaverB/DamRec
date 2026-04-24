# -*- coding: utf-8 -*-
# FroRecEps8: FroRec 超参探针子类，adam_eps=1e-8 (见 sequential_FroRecEps8.yaml)
# 通过独立 class 名让 RecBole 的 get_model + MODEL_CONFIGS 能按 key 区分同架构不同超参的变体。

from recbole.model.sequential_recommender.frorec import FroRec


class FroRecEps8(FroRec):
    """FroRec hyperparameter probe: adam_eps=1e-8 (standard Adam default)."""
    pass
