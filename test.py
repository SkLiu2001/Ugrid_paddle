import argparse
import shutil
import time
import typing
import os

from loguru import logger
import numpy as np
import paddle
import paddle.nn.functional as F

from arg import TestArg
from data import SynDat
from model import Solver
import util


# noinspection DuplicatedCode
def test_on_dataset(model, test_loader) -> None:
    model.eval()
    with paddle.no_grad():
        test_loss_dict: typing.Dict[str, typing.List[paddle.Tensor]] = {}

        for batch in test_loader:
            x: typing.Optional[paddle.Tensor] = None
            bc_value: paddle.Tensor = batch['bc_value']
            bc_mask: paddle.Tensor = batch['bc_mask']
            f: typing.Optional[paddle.Tensor] = None

            tup: typing.Tuple[paddle.Tensor, int] = model(x, bc_value, bc_mask, f)
            y, iterations_used = tup

            absolute_loss, relative_loss = util.relative_residue(y, bc_value, bc_mask, f)
            absolute_loss = paddle.mean(absolute_loss)
            relative_loss = paddle.mean(relative_loss)

            iterations_used = paddle.to_tensor([iterations_used], dtype='float32')

            if 'absolute_loss' in test_loss_dict:
                test_loss_dict['absolute_loss'].append(absolute_loss)
            else:
                test_loss_dict['absolute_loss']: typing.List[paddle.Tensor] = [absolute_loss]

            if 'relative_loss' in test_loss_dict:
                test_loss_dict['relative_loss'].append(relative_loss)
            else:
                test_loss_dict['relative_loss']: typing.List[paddle.Tensor] = [relative_loss]

            if 'iterations_used' in test_loss_dict:
                test_loss_dict['iterations_used'].append(iterations_used)
            else:
                test_loss_dict['iterations_used']: typing.List[paddle.Tensor] = [iterations_used]

        for k, v in test_loss_dict.items():
            # Stack tensors along a new axis
            stacked = paddle.stack(v)
            logger.info('[Test] {} = {}'.format(k, paddle.mean(stacked)))


# noinspection DuplicatedCode
def test_on_single_data(testcase: str,
                        size: int,
                        model: Solver,
                        benchmark_iteration: typing.Optional[int] = None) \
        -> None:
    model.eval()
    with paddle.no_grad():
        test_loss_dict: typing.Dict[str, typing.List[paddle.Tensor]] = {}

        image_size: int = size
        bc_value, bc_mask, f = util.get_testcase(testcase, image_size, 'gpu')  # Add device parameter

        time_lst: typing.List[float] = []

        if benchmark_iteration is None:
            benchmark_iteration = 1

        for _ in range(benchmark_iteration):
            start_time: float = time.perf_counter_ns()
            tup: typing.Tuple[paddle.Tensor, int] = model(None, bc_value, bc_mask, f)
            y, iterations_used = tup
            time_lst.append(time.perf_counter_ns() - start_time)

            # Check for numerical instability
            if paddle.isnan(y).any() or paddle.isinf(y).any():
                logger.warning(f'Model output contains NaN/Inf! Range: {paddle.min(y).item()} ~ {paddle.max(y).item()}')

        tup: typing.Tuple[paddle.Tensor, paddle.Tensor] = util.relative_residue(y, bc_value, bc_mask, f)
        abs_residual_norm, rel_residual_norm = tup
        abs_residual_norm: paddle.Tensor = paddle.mean(abs_residual_norm)
        rel_residual_norm: paddle.Tensor = paddle.mean(rel_residual_norm)

        iterations_used = paddle.to_tensor([iterations_used], dtype='float32')

        if 'abs_residual_norm' in test_loss_dict:
            test_loss_dict['abs_residual_norm'].append(abs_residual_norm)
        else:
            test_loss_dict['abs_residual_norm']: typing.List[paddle.Tensor] = [abs_residual_norm]

        if 'rel_residual_norm' in test_loss_dict:
            test_loss_dict['rel_residual_norm'].append(rel_residual_norm)
        else:
            test_loss_dict['rel_residual_norm']: typing.List[paddle.Tensor] = [rel_residual_norm]

        if 'iterations_used' in test_loss_dict:
            test_loss_dict['iterations_used'].append(iterations_used)
        else:
            test_loss_dict['iterations_used']: typing.List[paddle.Tensor] = [iterations_used]

        bc_value_np: np.ndarray = bc_value.numpy().squeeze()
        bc_mask_np: np.ndarray = bc_mask.numpy().squeeze()
        f_np: typing.Optional[np.ndarray] = f.numpy().squeeze() if f is not None else None
        y_np: np.ndarray = y.numpy().squeeze()

        # Use paddle.stack instead of paddle.concat for single tensor
        rel_residual_norm_mean = paddle.mean(test_loss_dict['rel_residual_norm'][0])
        
        log_str = (f'UGrid {testcase}_{size} {np.mean(time_lst) / 1e6} ms, ' +
                   'rel res {}'.format(rel_residual_norm_mean.item()))
        logger.info(log_str)


# noinspection DuplicatedCode
def main() -> None:
    # argument parameters
    arg_opt: argparse.Namespace = TestArg().parse()

    # training parameters
    experiment_checkpoint_path: str = os.path.join(arg_opt.checkpoint_root, arg_opt.load_experiment)
    exp_opt_np: np.ndarray = np.load(os.path.join(experiment_checkpoint_path, 'opt_old.npy'), allow_pickle=True)

    # merged argument namespace
    opt: argparse.Namespace = util.merge_namespace(exp_opt_np.item(), arg_opt)

    logger.info('======================== Args ========================')
    for k, v in vars(opt).items():
        logger.info(f'{k}\t\t{v}')
    logger.info('======================================================\n')

    if opt.seed is not None:
        paddle.seed(opt.seed)
        logger.info(f'[Test] Manual seed PaddlePaddle with seed {opt.seed}\n')
    else:
        seed: int = int(paddle.get_cuda_rng_state()[0])
        paddle.seed(seed)
        logger.info(f'[Test] Using random seed {seed} for PaddlePaddle\n')

    # model
    opt.num_iterations = 64
    model = Solver(opt.structure, opt.downsampling_policy, opt.upsampling_policy,
                   'gpu',  # device
                   opt.num_iterations, 1e-4, opt.initialize_x0,
                   opt.num_mg_layers, opt.num_mg_pre_smoothing, opt.num_mg_post_smoothing,
                   opt.activation, 'default')  # initialize_trainable_parameters
    
    loaded_checkpoint = model.load(experiment_checkpoint_path, opt.load_epoch)
    model.eval()
    logger.info(f'[Test] Checkpoint loaded from {loaded_checkpoint}\n')

    testcase_lst: typing.List[str] = ['bag', 'cat', 'lock', 'note', 'poisson_region', 'punched_curve',
                                      'shape_l', 'shape_square', 'shape_square_poisson', 'star', 'bag']

    # Test UGrid
    for size in [4097]:
        for testcase in testcase_lst:
            test_on_single_data(testcase, size, model, benchmark_iteration=10)


if __name__ == '__main__':
    main()








