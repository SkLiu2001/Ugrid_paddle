import typing
import os

import paddle
import paddle.io

from data import SynDat
import util


class Trainer:
    def __init__(self,
                 experienment_name: str, experienment_checkpoint_path: str, device: str,
                 model, logger,
                 optimizer: str, scheduler: str, initial_lr: float, lambda_1: float, lambda_2: float,
                 start_epoch: int, max_epoch: int, save_every: int, evaluate_every: int,
                 dataset_root: str, num_workers: int, batch_size: int,):
        self.experienment_name: str = experienment_name
        self.experienment_checkpoint_path: str = experienment_checkpoint_path

        self.model = model
        self.logger = logger

        self.device: str = device

        self.initial_lr: float = initial_lr
        self.lambda_1: float = lambda_1
        self.lambda_2: float = lambda_2

        self.start_epoch: int = start_epoch
        self.max_epoch: int = max_epoch
        self.save_every: int = save_every
        self.evaluate_every: int = evaluate_every

        self.dataset_root: str = dataset_root
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

        if optimizer == 'adam':
            self.optimizer = paddle.optimizer.Adam(learning_rate=self.initial_lr, parameters=self.model.parameters())
        elif optimizer == 'rmsprop':
            self.optimizer = paddle.optimizer.RMSProp(learning_rate=self.initial_lr, parameters=self.model.parameters())
        elif optimizer == 'sgd':
            self.optimizer = paddle.optimizer.SGD(learning_rate=self.initial_lr, parameters=self.model.parameters())
        else:
            raise NotImplementedError

        if 'step' in scheduler:
            _, step_size, gamma = scheduler
            self.scheduler = paddle.optimizer.lr.StepDecay(learning_rate=self.initial_lr, step_size=int(step_size), gamma=float(gamma))
            # Update optimizer to use scheduler
            if optimizer == 'adam':
                self.optimizer = paddle.optimizer.Adam(learning_rate=self.scheduler, parameters=self.model.parameters())
            elif optimizer == 'rmsprop':
                self.optimizer = paddle.optimizer.RMSProp(learning_rate=self.scheduler, parameters=self.model.parameters())
            elif optimizer == 'sgd':
                self.optimizer = paddle.optimizer.SGD(learning_rate=self.scheduler, parameters=self.model.parameters())
        else:
            raise NotImplementedError

        train_dataset_path: str = os.path.join(dataset_root, 'train')
        self.train_dataset = SynDat(train_dataset_path)
        self.train_loader = paddle.io.DataLoader(self.train_dataset,
                                                 num_workers=self.num_workers,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        logger.info(f'[Trainer] {len(self.train_dataset)} training data loaded from {train_dataset_path}')

        evaluate_dataset_path: str = os.path.join(dataset_root, 'evaluate')
        self.evaluate_dataset = SynDat(evaluate_dataset_path)
        self.evaluate_loader = paddle.io.DataLoader(self.evaluate_dataset,
                                                    num_workers=self.num_workers,
                                                    batch_size=batch_size)

        logger.info(f'[Trainer] {len(self.evaluate_dataset)} evaluation data loaded from {evaluate_dataset_path}\n')

    # noinspection DuplicatedCode
    def train(self):
        # Set gradient clipping
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        
        for epoch in range(self.start_epoch, self.max_epoch):
            # Train
            self.model.train()
            train_loss_dict: typing.Dict[str, typing.List[paddle.Tensor]] = {}

            for batch in self.train_loader:
                x: typing.Optional[paddle.Tensor] = None
                bc_value: paddle.Tensor = batch['bc_value']
                bc_mask: paddle.Tensor = batch['bc_mask']
                f: typing.Optional[paddle.Tensor] = None

                if self.device == 'gpu':
                    bc_value = bc_value.cuda()
                    bc_mask = bc_mask.cuda()

                # Debug input shapes
                #print(f"\n[Debug] Input shapes: bc_value {bc_value.shape}, bc_mask {bc_mask.shape}")
                #print(f"[Debug] Input ranges: bc_value [{paddle.min(bc_value)}, {paddle.max(bc_value)}], bc_mask [{paddle.min(bc_mask)}, {paddle.max(bc_mask)}]")

                tup: typing.Tuple[paddle.Tensor, int] = self.model(x, bc_value, bc_mask, f)
                y, iterations_used = tup

                # Scale down model output if too large (lower threshold)
                if paddle.max(paddle.abs(y)) > 1e5:
                    scale_factor = 1e5 / paddle.max(paddle.abs(y))
                    y = y * scale_factor
                    print(f"[Debug] Scaled down model output by factor {scale_factor}")

                # Debug model output
                print(f"[Debug] Model output range: [{paddle.min(y)}, {paddle.max(y)}]")

                residue: paddle.Tensor = util.absolute_residue(y, bc_mask, f, reduction='none')

                # Scale down residue if too large (lower threshold)
                if paddle.max(paddle.abs(residue)) > 1e5:
                    scale_factor = 1e5 / paddle.max(paddle.abs(residue))
                    residue = residue * scale_factor
                    print(f"[Debug] Scaled down residue by factor {scale_factor}")

                # Debug residue
                print(f"[Debug] Residue range: [{paddle.min(residue)}, {paddle.max(residue)}]")

                # Apply log1p to make the loss more stable
                loss_x: paddle.Tensor = paddle.log1p(util.norm(residue)).mean()

                if paddle.any(paddle.isnan(loss_x)) or paddle.any(paddle.isinf(loss_x)):
                    print(f"[Warning] loss_x is NaN/Inf! Value: {loss_x}")
                    continue  # Skip this batch if loss is invalid

                iterations_used = paddle.to_tensor([iterations_used], dtype='float32')
                if self.device == 'gpu':
                    iterations_used = iterations_used.cuda()

                if 'loss_x' in train_loss_dict:
                    train_loss_dict['loss_x'].append(loss_x)
                else:
                    train_loss_dict['loss_x']: typing.List[paddle.Tensor] = [loss_x]

                if 'iterations_used' in train_loss_dict:
                    train_loss_dict['iterations_used'].append(iterations_used)
                else:
                    train_loss_dict['iterations_used']: typing.List[paddle.Tensor] = [iterations_used]

                loss = self.lambda_1 * loss_x

                if paddle.any(paddle.isnan(loss)) or paddle.any(paddle.isinf(loss)):
                    print(f"[Warning] Final loss is NaN/Inf! Value: {loss}")
                    continue  # Skip this batch if loss is invalid

                if 'loss' in train_loss_dict:
                    train_loss_dict['loss'].append(loss)
                else:
                    train_loss_dict['loss']: typing.List[paddle.Tensor] = [loss]

                self.optimizer.clear_grad()
                loss.backward()
                
                # Apply gradient clipping
                params_grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        params_grads.append((param, param.grad))
                grad_clip(params_grads)
                
                self.optimizer.step()

            for k, v in train_loss_dict.items():
                self.logger.info('[Epoch {}/{}] {} = {}'.format(epoch, self.max_epoch - 1,
                                                                k, paddle.mean(paddle.stack(v))))

            # Evaluate
            if 0 < self.evaluate_every and (epoch + 1) % self.evaluate_every == 0:
                self.model.eval()
                evaluate_loss_dict: typing.Dict[str, typing.List[paddle.Tensor]] = {}

                for batch in self.evaluate_loader:
                    x: typing.Optional[paddle.Tensor] = None
                    bc_value: paddle.Tensor = batch['bc_value']
                    bc_mask: paddle.Tensor = batch['bc_mask']
                    f: typing.Optional[paddle.Tensor] = None

                    if self.device == 'gpu':
                        bc_value = bc_value.cuda()
                        bc_mask = bc_mask.cuda()

                    tup: typing.Tuple[paddle.Tensor, int] = self.model(x, bc_value, bc_mask, f)
                    y, iterations_used = tup

                    abs_residual_norm, rel_residual_norm = util.relative_residue(y, bc_value, bc_mask, f)
                    abs_residual_norm = abs_residual_norm.mean()
                    rel_residual_norm = rel_residual_norm.mean()

                    if 'abs_residual_norm' in evaluate_loss_dict:
                        evaluate_loss_dict['abs_residual_norm'].append(abs_residual_norm)
                    else:
                        evaluate_loss_dict['abs_residual_norm']: typing.List[paddle.Tensor] = [abs_residual_norm]

                    if 'rel_residual_norm' in evaluate_loss_dict:
                        evaluate_loss_dict['rel_residual_norm'].append(rel_residual_norm)
                    else:
                        evaluate_loss_dict['rel_residual_norm']: typing.List[paddle.Tensor] = [rel_residual_norm]

                for k, v in evaluate_loss_dict.items():
                    self.logger.info('[Evaluation] {} = {}'.format(k, paddle.mean(paddle.stack(v))))

                self.model.train()

            # Scheduler step
            if hasattr(self, 'scheduler'):
                self.logger.info('[Epoch {}/{}] Current learning rate = {}'.format(epoch,
                                                                                   self.max_epoch - 1,
                                                                                   self.scheduler.get_lr()))
                self.scheduler.step()
            else:
                self.logger.info('[Epoch {}/{}] Current learning rate = {}'.format(epoch,
                                                                                   self.max_epoch - 1,
                                                                                   self.optimizer.get_lr()))

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or epoch == self.max_epoch - 1:
                self.logger.info('[Epoch {}/{}] Model saved.\n'.format(epoch, self.max_epoch - 1))
                self.model.save(self.experienment_checkpoint_path, epoch + 1)
            else:
                self.logger.info('')
