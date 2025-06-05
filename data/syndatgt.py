from .syndat import SynDat

import numpy as np
import paddle


class SynDatGt(SynDat):
    def __init__(self, dataset_root: str):
        super().__init__(dataset_root)

    def __getitem__(self, index: int):
        data: np.ndarray = np.load(self.instance_path_lst[index])
        bc_value, bc_mask, gt = data
        bc_value: paddle.Tensor = paddle.to_tensor(bc_value).astype('float32').unsqueeze(0)
        bc_mask: paddle.Tensor = paddle.to_tensor(bc_mask).astype('float32').unsqueeze(0)
        gt: paddle.Tensor = paddle.to_tensor(gt).astype('float32').unsqueeze(0)
        return {'bc_value': bc_value,
                'bc_mask':  bc_mask,
                'gt':       gt}

    def __len__(self):
        return len(self.instance_path_lst) // 4
