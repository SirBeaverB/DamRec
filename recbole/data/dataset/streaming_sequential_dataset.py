# -*- coding: utf-8 -*-
# StreamingSequentialDataset: 在 augmentation 前保存原始 inter_feat，用于构建全局时间轴

from recbole.data.interaction import Interaction
from recbole.data.dataset.sequential_dataset import SequentialDataset


class StreamingSequentialDataset(SequentialDataset):
    """SequentialDataset 子类，当 streaming_t2t=True 时在 augmentation 前保存原始数据供 timeline 构建。"""

    def data_augmentation(self):
        if self.config.final_config_dict.get("streaming_t2t", False):
            raw_dict = {k: v.clone() for k, v in self.inter_feat.interaction.items()}
            self._raw_inter_for_timeline = Interaction(raw_dict)
        super().data_augmentation()
