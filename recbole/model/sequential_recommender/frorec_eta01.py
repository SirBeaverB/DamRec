# -*- coding: utf-8 -*-
# FroRecEta01: FroRec 超参探针子类，adam_eta=0.1 (见 sequential_FroRecEta01.yaml)

from recbole.model.sequential_recommender.frorec import FroRec


class FroRecEta01(FroRec):
    """FroRec hyperparameter probe: adam_eta=0.1 (double the streaming update magnitude)."""
    pass
