from torch import optim
import numpy as np

class CosineWarmupScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch+1) * 1.0 / (self.warmup+1)
        return lr_factor
    
class PolyLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_step, warmup_step, power=2.0,
                 warmup_factor=1/4, warmup_method='linear', target_lr=1e-8, last_epoch=-1):
        
        self.power = power
        self.max_step = max_step
        self.warmup_step = warmup_step
        self.warmup_factor = warmup_factor
        self.warmup_method = warmup_method
        self.target_lr = target_lr

        if self.warmup_method not in ('constant', 'linear'):
            raise ValueError(f"Invalid warmup_method: {self.warmup_method}")

        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _get_warmup_factor(method, iter, warmup_steps, warmup_factor):
        if iter >= warmup_steps:
            return 1.0
        if method == 'constant':
            return warmup_factor
        elif method == 'linear':
            alpha = iter / warmup_steps
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError(f"Unknown warmup method: {method}")

    def get_lr(self):
        N = self.max_step - self.warmup_step
        T = self.last_epoch - self.warmup_step

        if self.last_epoch <= self.warmup_step:
            warmup_factor = self._get_warmup_factor(
                self.warmup_method, self.last_epoch, self.warmup_step, self.warmup_factor
            )
            return [
                self.target_lr + (base_lr - self.target_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            factor = pow(1 - T / N, self.power)
            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]
