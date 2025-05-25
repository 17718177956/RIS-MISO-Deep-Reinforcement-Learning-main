import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np


# Implementation of the Deep Deterministic Policy Gradient algorithm (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, device, max_action=1):
        super(Actor, self).__init__()
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()
        self.device = device

        self.M = M
        self.N = N
        self.K = K
        self.power_t = power_t
        print(f"[DDPG] M={M}, N={N}, K={K}, power_t={power_t}")


        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.max_action = max_action

    def compute_power(self, a):
        # Normalize the power
        G_real = a[:, :self.M ** 2].cpu().data.numpy()
        G_imag = a[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()

        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)

        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))

        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)

        return current_power_t

    def compute_phase(self, a):
        # Normalize the phase matrix
        Phi_real = a[:, -2 * self.N:-self.N].detach()
        Phi_imag = a[:, -self.N:].detach()

        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)

    def forward(self, state):
        a = torch.tanh(self.l1(state.float()))

        # Apply batch normalization to the each hidden layer's input
        a = self.bn1(a)
        a = torch.tanh(self.l2(a))

        a = self.bn2(a)
        a = torch.tanh(self.l3(a))

        # Normalize the transmission power and phase matrix
        current_power_t = self.compute_power(a.detach()).expand(-1, 2 * self.M ** 2) / np.sqrt(self.power_t)

        real_normal, imag_normal = self.compute_phase(a.detach())

        real_normal = real_normal.expand(-1, self.N)
        imag_normal = imag_normal.expand(-1, self.N)
        ones_for_a = torch.ones(a.size(0), 2 * self.K).to(self.device)  # a 不做标准化
        division_term = torch.cat([current_power_t, real_normal, imag_normal, ones_for_a], dim=1)

        return self.max_action * a / division_term


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action):
        q = torch.tanh(self.l1(state.float()))

        q = torch.tanh(self.l2(torch.cat([q, action], 1)))

        q = self.l3(q)

        return q


class DDPG(object):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, max_action,
                 actor_lr, critic_lr, actor_decay, critic_decay, device,
                 discount=0.90, tau=0.001,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.policy_freq = policy_freq  
        self.device = device
        # --- ① 以子文件夹区分两条 Critic 的日志 --------------------
        self.writer      = SummaryWriter("runs/debug")           # 全局
        self.writer_c1   = SummaryWriter("runs/debug/critic1")   # Critic-1
        self.writer_c2   = SummaryWriter("runs/debug/critic2")   # Critic-2
        self.total_it = 0  
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq


        powert_t_W = 10 ** (power_t / 10)

        # Initialize actor networks and optimizer
        self.actor = Actor(state_dim, action_dim, M, N, K, powert_t_W, max_action=max_action, device=device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay)

        # 修改后：添加第二个 Critic
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target1 = copy.deepcopy(self.critic1)
        self.critic_target2 = copy.deepcopy(self.critic2)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=critic_decay)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=critic_decay)
        # Initialize the discount and target update rated
        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        self.actor.eval()

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten().reshape(1, -1)

        return action

    def update_parameters(self, replay_buffer, batch_size=16):
        self.actor.train()
        self.total_it += 1  # 更新计数器

        # Sample from the experience replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            # 计算两个目标 Critic 的 Q 值并取最小值:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}
            target_Q1 = self.critic_target1(next_state, next_action)
            target_Q2 = self.critic_target2(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)
        # ---------- ② 分别计算两个 loss -----------------------------
        q1_pred = self.critic1(state, action)
        q2_pred = self.critic2(state, action)
        loss_c1 = F.mse_loss(q1_pred, target_Q)
        loss_c2 = F.mse_loss(q2_pred, target_Q)

        # ---------- ③ 独立反向传播（两次 backward）-------------------
        # -- Critic-1
        self.critic_optimizer1.zero_grad()
        loss_c1.backward()
        grad_norm_c1 = torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5).item()
        self.critic_optimizer1.step()

        # -- Critic-2
        self.critic_optimizer2.zero_grad()
        loss_c2.backward()
        grad_norm_c2 = torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5).item()
        self.critic_optimizer2.step()

        # ---------- ④ 写 TensorBoard：两条折线 + 可选直方图 ----------
        if self.total_it % 100 == 0:                # 频率可调
            self.writer_c1.add_scalar("loss",       loss_c1.item(),  self.total_it)
            self.writer_c1.add_scalar("grad_norm",  grad_norm_c1,    self.total_it)
            self.writer_c2.add_scalar("loss",       loss_c2.item(),  self.total_it)            
            self.writer_c2.add_scalar("grad_norm",  grad_norm_c2,    self.total_it)

            # （可选）梯度分布直方图——隔 1 K 步记录一次即可
            if self.total_it % 1000 == 0:
                for n,p in self.critic1.named_parameters():
                    if p.grad is not None:
                        self.writer_c1.add_histogram(f"grad/{n}", p.grad, self.total_it)
                for n,p in self.critic2.named_parameters():
                    if p.grad is not None:
                        self.writer_c2.add_histogram(f"grad/{n}", p.grad, self.total_it)

        # 仅当达到延迟更新频率时更新 Actor 和目标网络:contentReference[oaicite:15]{index=15}
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            # 软更新所有目标网络
            for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




