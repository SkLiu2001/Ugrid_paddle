import argparse
import datetime
import os

import numpy as np
from loguru import logger
import paddle
import paddle.nn as nn

from arg import TrainArg
import model
import util


# noinspection DuplicatedCode
def main() -> None:
    # args
    opt: argparse.Namespace = TrainArg().parse()

    # checkpoint
    experienment_name: str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + 'paddle'
    experienment_checkpoint_path: str = os.path.join(opt.checkpoint_root, experienment_name)
    os.makedirs(experienment_checkpoint_path, exist_ok=True)
    np.save(os.path.join(experienment_checkpoint_path, 'opt_old.npy'), opt)

    # logger
    logger.add(os.path.join(experienment_checkpoint_path, 'train.log'))
    logger.info('======================== Args ========================')
    for k, v in vars(opt).items():
        logger.info(f'{k}\t\t{v}')
    logger.info('======================================================\n')

    # backend
    device = util.get_device()
    logger.info(f'[Train] Using device {device}')

    if opt.deterministic:
        # Set paddle to deterministic mode
        paddle.seed(opt.seed if opt.seed is not None else 42)
        logger.info(f'[Train] Enforce deterministic algorithms')
    else:
        logger.info(f'[Train] Do not enforce deterministic algorithms')

    if opt.seed is not None:
        paddle.seed(opt.seed)
        logger.info(f'[Train] Manual seed PaddlePaddle with seed {opt.seed}\n')
    else:
        seed: int = int(paddle.get_cuda_rng_state()[0])
        paddle.seed(seed)
        logger.info(f'[Train] Using random seed {seed} for PaddlePaddle\n')

    # paddle.set_grad_enabled(True)  # for debugging only

    # training
    solver = model.Solver(opt.structure, opt.downsampling_policy, opt.upsampling_policy, device,
                          opt.num_iterations, opt.relative_tolerance, opt.initialize_x0,
                          opt.num_mg_layers, opt.num_mg_pre_smoothing, opt.num_mg_post_smoothing,
                          opt.activation, opt.initialize_trainable_parameters)
    trainer = model.Trainer(experienment_name, experienment_checkpoint_path, device,
                            solver, logger,
                            opt.optimizer, opt.scheduler, opt.initial_lr, opt.lambda_1, opt.lambda_2,
                            opt.start_epoch, opt.max_epoch, opt.save_every, opt.evaluate_every,
                            opt.dataset_root, opt.num_workers, opt.batch_size)
    trainer.train()


if __name__ == '__main__':
    main()
