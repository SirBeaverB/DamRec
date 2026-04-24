# -*- coding: utf-8 -*-
# FroRecB999: FroRec 超参探针子类，adam_beta2=0.999 (见 sequential_FroRecB999.yaml)

from recbole.model.sequential_recommender.frorec import FroRec


class FroRecB999(FroRec):
    """FroRec hyperparameter probe: adam_beta2=0.999 (longer V EMA horizon)."""
    pass
