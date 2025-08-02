import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
import math
import numpy as np


class SAI_FGSM(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(SAI_FGSM, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):

        T = 10
        initValue = T
        Tmin = 0.5
        t = 1

        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        while T >= Tmin:
            for _ in range(self.total_step):
                # x.requires_grad = True
                # logit = 0
                # for model in self.models:
                #     logit += model(x.to(model.device)).to(x.device)
                # loss = self.criterion(logit, y)
                # loss.backward()
                # grad = x.grad
                grad = self.calculate_v(x, y)
                x.requires_grad = False
                # 这个地方不能一个batch整体计算norm。这样的话，没有机会出现norm为0的情况，应该一个样本一个样本计算
                if torch.norm(grad, p=2) != 0:
                    # update
                    if self.targerted_attack:
                        momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                        x += self.step_size * momentum.sign()
                    else:
                        momentum = self.mu * momentum * 1.5 + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) * 0.5
                        x += self.step_size * momentum.sign()
                else:
                    P = math.exp(-(1/T))
                    R = np.random.uniform(low=0, high=1)
                    if R < P:
                        if torch.norm(momentum, p=2) != 0:
                            x -= self.step_size * momentum
                        else:
                            x = self.perturb(x)

                # x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

            t = t + 1
            T = initValue / t

        return x


    def calculate_v(self, x: torch.tensor, y: torch.tensor, N=20, beta=1.5):
        """

        :param x:  B, C, H, D
        :param y:
        :param N:
        :param beta:
        :return:
        """
        B, C, H, D = x.shape
        x = x.reshape(1, B, C, H, D)
        x = x.repeat(N, 1, 1, 1, 1)  # N, B, C, H, D
        ranges = beta * self.epsilon
        now = x + (torch.rand_like(x) - 0.5) * 2 * ranges
        now = now.view(N * B, C, H, D)
        now.requires_grad = True
        logit = 0
        for model in self.models:
            logit += model(now.to(model.device)).to(now.device)
        loss = self.criterion(logit, y.repeat(N))
        loss.backward()
        v = now.grad.view(N, B, C, H, D)  # N, B, C, H, D
        v = v.mean(0)
        return v



class SAI_FGSM_increment(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(SAI_FGSM_increment, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):

        T = 10
        initValue = T
        Tmin = 0.5
        t = 1

        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        while T >= Tmin:
            for _ in range(self.total_step):
                x.requires_grad = True
                logit = 0
                for model in self.models:
                    logit += model(x.to(model.device)).to(x.device)
                loss = self.criterion(logit, y)
                loss.backward()
                grad = x.grad
                # grad = self.calculate_v(x, y)
                x.requires_grad = False
                # 这个地方不能一个batch整体计算norm。这样的话，没有机会出现norm为0的情况，应该一个样本一个样本计算
                if torch.norm(grad, p=2) != 0:
                    # update
                    if self.targerted_attack:
                        momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                        x += self.step_size * momentum.sign()
                    else:
                        momentum = self.mu * momentum * 1.5 + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) * 0.5
                        x += self.step_size * momentum.sign()
                else:
                    P = math.exp(-(1/T))
                    R = np.random.uniform(low=0, high=1)
                    if R < P:
                        if torch.norm(momentum, p=2) != 0:
                            x -= self.step_size * momentum
                        else:
                            x = self.perturb(x)

                # x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

            t = t + 1
            T = initValue / t

        return x


    def calculate_v(self, x: torch.tensor, y: torch.tensor, N=20, beta=1.5):
        """

        :param x:  B, C, H, D
        :param y:
        :param N:
        :param beta:
        :return:
        """
        B, C, H, D = x.shape
        x = x.reshape(1, B, C, H, D)
        x = x.repeat(N, 1, 1, 1, 1)  # N, B, C, H, D
        ranges = beta * self.epsilon
        now = x + (torch.rand_like(x) - 0.5) * 2 * ranges
        now = now.view(N * B, C, H, D)
        now.requires_grad = True
        logit = 0
        for model in self.models:
            logit += model(now.to(model.device)).to(now.device)
        loss = self.criterion(logit, y.repeat(N))
        loss.backward()
        v = now.grad.view(N, B, C, H, D)  # N, B, C, H, D
        v = v.mean(0)
        return v
