import os
import typing

import numpy as np
import paddle

from .ugrid import UGrid
import util


class Solver:
    def __init__(self,
                 structure: str,
                 downsampling_policy: str,
                 upsampling_policy: str,
                 device: str,
                 num_iterations: int,
                 relative_tolerance: float,
                 initialize_x0: str,
                 num_mg_layers: int,
                 num_mg_pre_smoothing: int,
                 num_mg_post_smoothing: int,
                 activation: str,
                 initialize_trainable_parameters: str):

        self.structure: str = structure
        self.device: str = device
        self.num_iterations: int = num_iterations
        self.initialize_x0: str = initialize_x0
        self.relative_tolerance: float = relative_tolerance
        self.initial_guess = lambda bc_value, bc_mask: util.initial_guess(bc_value, bc_mask, 'random')

        self.is_train: bool = True

        if self.structure == 'unet':
            self.iterator = UGrid(num_mg_layers,
                                  num_mg_pre_smoothing,
                                  num_mg_post_smoothing,
                                  downsampling_policy,
                                  upsampling_policy,
                                  activation,
                                  initialize_trainable_parameters)
            if self.device == 'gpu':
                paddle.device.set_device('gpu')
        else:
            raise NotImplementedError

    def __call__(self,
                 x: typing.Optional[paddle.Tensor],
                 bc_value: paddle.Tensor,
                 bc_mask: paddle.Tensor,
                 f: typing.Optional[paddle.Tensor],
                 rel_tol: typing.Optional[float] = None) \
            -> typing.Tuple[paddle.Tensor, int]:
        if not self.is_train:
            if rel_tol is None:
                rel_tol: float = self.relative_tolerance

            rhs = bc_value

            if f is not None:
                rhs = rhs + f

            rhs_norm: paddle.Tensor = util.norm(rhs)
            abs_tol: paddle.Tensor = rel_tol * rhs_norm

        if x is None:
            x: paddle.Tensor = self.initial_guess(bc_value, bc_mask)

        # # TODO: UGrid benchmark
        # np.save(f'var/conv/UGrid/tmp/x0.npy',
        #         x.detach().squeeze().cpu().numpy())

        for iteration in range(1, self.num_iterations + 1):
            x: paddle.Tensor = self.iterator(x, bc_value, bc_mask, f)

            # # TODO: UGrid benchmark
            # np.save(f'var/conv/UGrid/tmp/x{iteration}.npy',
            #         x.detach().squeeze().cpu().numpy())

            if not self.is_train:
                if iteration % 4 == 0 and \
                        paddle.all(util.absolute_residue(x, bc_mask, f, reduction='norm') <= abs_tol):
                    break

        # noinspection PyUnboundLocalVariable
        return x, iteration

    def train(self):
        self.is_train = True
        self.iterator.train()

    def eval(self):
        self.is_train = False
        self.iterator.eval()

    def parameters(self):
        return self.iterator.parameters()

    def load(self, checkpoint_path: str, epoch: int):
        checkpoint_pth_root: str = os.path.join(checkpoint_path, 'pth')

        if epoch == -1:
            epoch = util.get_number_of_files(checkpoint_pth_root)

        load_path: str = os.path.join(checkpoint_path, 'pth', f'epoch_{epoch}.pth')
        state_dict = paddle.load(load_path)
        self.iterator.set_state_dict(state_dict)

        return load_path

    def save(self, checkpoint_path: str, epoch: int):
        save_dir: str = os.path.join(checkpoint_path, 'pth')
        os.makedirs(save_dir, exist_ok=True)
        save_path: str = os.path.join(save_dir, f'epoch_{epoch}.pth')
        paddle.save(self.iterator.state_dict(), save_path)
