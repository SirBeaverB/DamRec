# -*- coding: utf-8 -*-
# DamRec: Delta-Adam Memory for Streaming Recommendation
#
# Extends GDN by replacing SGD-equivalent state update with Adam-equivalent.
# TODO: Implement Adam-equivalent delta rule (momentum + adaptive scaling).

from recbole.model.sequential_recommender.gdn import GDN


class DamRec(GDN):
    r"""DamRec: Delta-Adam Memory for Streaming Recommendation.

    Extends GDN by upgrading the state update from SGD-equivalent to Adam-equivalent:
    - GDN: h_t = gate * h_{t-1} + (1-gate) * delta  (SGD)
    - DamRec: Adam-style momentum + adaptive scaling in the delta rule

    Currently inherits GDN; Adam-equivalent _gated_delta_step to be implemented.
    """

    def __init__(self, config, dataset):
        super(DamRec, self).__init__(config, dataset)
        # TODO: Add Adam-specific params (beta1, beta2, eps) and buffers (m, v)

    # TODO: Override _gated_delta_step with Adam-equivalent update
    # def _gated_delta_step(self, x, prev_h=None):
    #     # Adam: m_t = beta1*m_{t-1} + (1-beta1)*delta
    #     #       v_t = beta2*v_{t-1} + (1-beta2)*delta^2
    #     #       h_t = gate * h_{t-1} + (1-gate) * m_t / (sqrt(v_t) + eps)
    #     ...
