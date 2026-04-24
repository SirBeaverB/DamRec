# -*- coding: utf-8 -*-
# FroRecNoV: Ablation of FroRec with the scalar second-moment V disabled.
#
# Keeps the first-order momentum M + Adam bias correction; forces denom = 1 so the
# update reduces to S_next = S_end + η·(M / bc1). Used to isolate whether the
# scalar Frobenius preconditioner contributes independently of momentum.

from recbole.model.sequential_recommender.frorec import FroRec


class FroRecNoV(FroRec):
    """FroRec without the second-order scalar V (M + bc1 only)."""

    def __init__(self, config, dataset):
        self.use_first_moment = True
        self.use_second_moment = False
        super().__init__(config, dataset)
