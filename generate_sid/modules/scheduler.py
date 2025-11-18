import torch
import numpy as np


class WarmUpAndCosineDecayScheduler:
    def __init__(self, optimizer, start_lr, base_lr, final_lr,
                 epoch_num, warmup_epoch_num):
        self.optimizer = optimizer
        self.step_counter = 0
        warmup_step_num = warmup_epoch_num
        decay_step_num = (epoch_num - warmup_epoch_num)
        warmup_lr_schedule = np.linspace(start_lr, base_lr, warmup_step_num)
        cosine_lr_schedule = final_lr + 0.5 * \
            (base_lr - final_lr) * (1 + np.cos(np.pi *
                                               np.arange(decay_step_num) / decay_step_num))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # step at each mini-batch
    def step(self):
        curr_lr = self.lr_schedule[self.step_counter]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = curr_lr
        self.step_counter += 1
        return curr_lr

    def state_dict(self):
        r"""
        It contains an entry for every variable in self.__dict__
        which is not one of the ('optimizer', 'lr_scheduler').

        Returns:
            the state of the scheduler as a dict.
        """
        return {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}

    def load_state_dict(self, state_dict):
        r"""
        Loads the schedulers state.

        Args:
            state_dict: dict = scheduler state. Should be an object returned from a call to state_dict.
        """
        self.__dict__.update(state_dict)

