# https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

import math
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
        Cosine Annealing:

        This adjusts the learning rate using a cosine function, where the learning rate decreases gradually
         from a maximum value to a minimum value over the course of a cycle. T
         his approach can help in fine-tuning the model as it approaches convergence.
        Warm-up:
        During the initial phase (warm-up steps), the learning rate starts from a minimum value and increases linearly
        to the maximum learning rate. This is useful to prevent large updates at the beginning of training,
        which can destabilize the training process.
        Restarts:
        After each cycle, the learning rate schedule restarts.
        The length of the cycles can increase geometrically
        (e.g., the number of steps in the next cycle might be a multiple of the previous cycle).
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        '''
        optimizer: The optimizer for which this scheduler will adjust the learning rates.
        first_cycle_steps: Number of steps in the first cycle.
        cycle_mult: Magnification factor for the cycle length. If this is greater than 1, each subsequent cycle will be longer.
        max_lr: The maximum learning rate at the start of the first cycle.
        min_lr: The minimum learning rate.
        warmup_steps: Number of steps for the linear warm-up phase at the start of the first cycle.
        gamma: Rate at which the maximum learning rate decreases after each cycle.
        last_epoch: Index of the last epoch. Default is -1, which means no previous state.
        '''
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size/ Number of steps in the first cycle.
        self.cycle_mult = cycle_mult # cycle steps magnification/ Magnification factor for the cycle length.
        # I# f this is greater than 1, each subsequent cycle will be longer.
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle/ The maximum learning rate at the
        # start of the first cycle.
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size Number of steps for the linear warm-up phase
        # at the start of the first cycle.
        self.gamma = gamma # decrease rate of max learning rate by cycle
        #  Rate at which the maximum learning rate decreases after each cycle.
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        self.last_epoch = last_epoch
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # If self.step_in_cycle is less than warmup_steps, the learning rate linearly
            # increases from min_lr to max_lr.
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            # After the warm-up, the learning rate decreases following a cosine function from max_lr to min_lr.
            # This is calculated using:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            '''
            If no epoch is provided,
             it assumes you're calling step() at every iteration,
             updates step_in_cycle, and checks if it exceeds the current cycle length.
             This formula allows for flexible and progressively longer cycles.
              If cycle_mult > 1, each cycle becomes longer, allowing for more gradual learning rate decay 
              as training progresses. This can be beneficial because it gives the model more time to learn during later
               cycles when learning rates are smaller.

                If cycle_mult = 1: The cycle lengths remain constant, meaning each cycle (after the first) has the 
                same number of steps.
                If cycle_mult > 1: Each cycle is longer than the previous one, which could help the model settle 
                into better minima by giving it more time with smaller learning rates.
            '''
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                # it restarts the cycle and updates cur_cycle_steps for the next cycle based on cycle_mult.
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                # self.cur_cycle_steps: Represents the total number of steps in the current cycle,
                # including warm-up steps.
                # self.warmup_steps: The number of steps reserved for the warm-up phase.
                #After completing a cycle, the scheduler needs to determine the length of the next cycle.
                # This equation updates self.cur_cycle_steps to reflect the total number of steps in the next cycle.
                # This part calculates the number of steps in the current cycle, excluding the warm-up steps.
                # The reason for subtracting self.warmup_steps is to isolate the portion of the cycle where
                # cosine annealing occurs.
                # If cycle_mult > 1, the length of the next cycle will be longer than the current one.
                # Finally, the calculated length for the next cycle (which excludes warm-up) is combined
                # with the warm-up steps to get the total number of steps in the next cycle.
        else:
            # it calculates the current cycle and step within the cycle based on how many epochs have passed.
            # It adjusts cur_cycle_steps for the current cycle and updates the learning rate accordingly.
            # epoch: The current epoch or iteration count passed to the method.
            # This is used to determine where the scheduler is within the entire training process.
            # first_cycle_steps: The number of steps in the first cycle.
            if epoch >= self.first_cycle_steps:
                # When the epoch is greater than or equal to first_cycle_steps,
                # the logic must determine the current cycle and the current step within that cycle.
                # This is because the scheduler might be in a later cycle after one or more restarts
                # This is straightforward when cycle_mult == 1 because the cycle lengths are uniform.
                if self.cycle_mult == 1.:
                    # Means that all cycles have the same length
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    #  The step within the current cycle is simply
                    # the remainder when the epoch is divided by the cycle length.
                    self.cycle = epoch // self.first_cycle_steps
                    # The current cycle is determined by integer division of the epoch by the cycle length.
                else:# When cycle_mult > 1 (Increasing Cycle Lengths):
                    # n Calculation:
                    # This calculation determines how many cycles (n) have been completed by solving
                    # for n in a geometric series.
                    # Taking the logarithm and solving for n gives the number of completed cycles.
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    # After determining how many full cycles have been completed,
                    # the step within the current cycle is calculated by subtracting the total number
                    # of steps in all previous cycles from the current epoch.
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    # Finally, the total number of steps in the current cycle is updated to
                    # reflect the length of the n-th cycle, which is calculated as:
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                # If epoch is less than first_cycle_steps, we are still in the first cycle.
                # self.cur_cycle_steps = self.first_cycle_steps: The number of steps in
                # the cycle is simply first_cycle_steps.
                self.cur_cycle_steps = self.first_cycle_steps
                # self.step_in_cycle = epoch: The current step within this first cycle is just the epoch.
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


if __name__=='__main__':
    optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.1)
    # Initialize the scheduler
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=10,
                                              cycle_mult=2,
                                              max_lr=0.1,
                                              min_lr=0.001,
                                              warmup_steps=3,
                                              gamma=0.5)
    # you have gamma equal 0.5 so your learning rate is decreased every time
    # very interesting idea
    # Number of epochs to simulate
    num_epochs = 70

    # Track learning rates
    lrs = []

    for epoch in range(num_epochs):
        scheduler.step(epoch)
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot the learning rates
    plt.plot(range(num_epochs), lrs, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.show()