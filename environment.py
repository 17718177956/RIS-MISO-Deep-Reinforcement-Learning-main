import numpy as np



class RIS_MISO(object):
    def __init__(self,
                 num_antennas,
                 num_RIS_elements,
                 num_users,
                 seed: int = 0,
                 channel_est_error=False,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2):
        
        np.random.seed(seed)
        self.M = num_antennas
        self.L = num_RIS_elements
        self.K = num_users
        # 生成信道：H1（BS->RIS），H_bn/H_bf（BS->用户），H_rn/H_rf（RIS->用户）
        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.M))
        self.H_bn = np.random.normal(0, np.sqrt(0.5), (self.M, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.M, self.K))
        self.H_bf = np.random.normal(0, np.sqrt(0.2), (self.M, self.K)) + 1j * np.random.normal(0, np.sqrt(0.2), (self.M, self.K))
        self.H_rn = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))
        self.H_rf = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))

        self.channel_est_error = channel_est_error
        self.awgn_var = AWGN_var
        assert self.M == self.K
        power_size = 2 * self.K  # (K对用户的传输功率) + (K对用户的反射功率)
        channel_size = 2 * (self.L * self.M + 2 * self.M * self.K + 2 * self.L * self.K)
        # H1: L×M, H_bn: M×K, H_bf: M×K, H_rn: L×K, H_rf: L×K
        self.action_dim = 2 * self.M * self.K + 2 * self.K + 2 * self.L
        self.state_dim = power_size + channel_size + self.action_dim

        #self.H_1 = None
        #self.H_rn = None  # RIS到近用户的信道
        #self.H_rf = None  # RIS到远用户的信道
        #self.H_bn = None  # BS到近用户的信道
        #self.H_bf = None  # BS到远用户的信道
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.L, dtype=complex)

        self.state = None
        self.done = None

        self.episode_t = None

    def reset(self):
        self.episode_t = 0

        # 生成信道：H1（BS->RIS），H_bn/H_bf（BS->用户），H_rn/H_rf（RIS->用户）
        #self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.M))
        #self.H_bn = np.random.normal(0, np.sqrt(0.5), (self.M, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.M, self.K))
        #self.H_bf = np.random.normal(0, np.sqrt(0.2), (self.M, self.K)) + 1j * np.random.normal(0, np.sqrt(0.2), (self.M, self.K))
        #self.H_rn = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))
        #self.H_rf = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack(
            (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi))
        # 初始化功率分配系数 a_{k,n}=a_{k,f}=0.5 为所有 K 对用户
        init_a_n = np.ones((1, self.K)) * 0.5
        init_a_f = np.ones((1, self.K)) * 0.5
        init_action = np.hstack((init_action_G, init_a_n, init_a_f, init_action_Phi))

        Phi_real = init_action[:, -2 * self.L:-self.L]
        Phi_imag = init_action[:, -self.L:]

        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        # 直接计算 power_r，而不再使用 H_2_tilde
        power_r = np.linalg.norm(self.H_rf + self.H_rn, axis=0).reshape(1, -1) ** 2


        H1_real = np.real(self.H_1).reshape(1, -1); H1_imag = np.imag(self.H_1).reshape(1, -1)
        H_bn_real = np.real(self.H_bn).reshape(1, -1); H_bn_imag = np.imag(self.H_bn).reshape(1, -1)
        H_bf_real = np.real(self.H_bf).reshape(1, -1); H_bf_imag = np.imag(self.H_bf).reshape(1, -1)
        H_rn_real = np.real(self.H_rn).reshape(1, -1); H_rn_imag = np.imag(self.H_rn).reshape(1, -1)
        H_rf_real = np.real(self.H_rf).reshape(1, -1); H_rf_imag = np.imag(self.H_rf).reshape(1, -1)
        self.state = np.hstack((init_action, power_t, power_r,
                             H1_real, H1_imag,
                             H_bn_real, H_bn_imag,
                            H_bf_real, H_bf_imag,
                             H_rn_real, H_rn_imag,
                             H_rf_real, H_rf_imag))

        return self.state

    def _compute_reward(self, Phi):
        reward = 0
        opt_reward = 0

        for k in range(self.K):
        # 处理第 k 对用户的信道：
         h_bn_k = self.H_bn[:, k].reshape(1, -1)
         h_bf_k = self.H_bf[:, k].reshape(1, -1)
         h_rn_k = self.H_rn[:, k].reshape(1, -1)
         h_rf_k = self.H_rf[:, k].reshape(1, -1)
         g_k = self.G[:, k].reshape(-1, 1)
         # 计算有效的 BS->用户信道（直接信道 + RIS辅助信道）：
         c_n_k = h_bn_k + (h_rn_k @ Phi @ self.H_1)   # 1×M 行向量表示近用户
         c_f_k = h_bf_k + (h_rf_k @ Phi @ self.H_1)   # 1×M 行向量表示远用户
         # 对于近用户和远用户的信号功率：
         signal_n = c_n_k @ g_k
         signal_f = c_f_k @ g_k
         x_n = self.a_n[k] * (np.abs(signal_n) ** 2)
         x_f = self.a_f[k] * (np.abs(signal_f) ** 2)
         # 计算来自其他信道的干扰功率：
         G_removed = np.delete(self.G, k, axis=1)         # M×(K-1)
         interference_n = np.sum(np.abs(c_n_k @ G_removed) ** 2)
         interference_f = np.sum(np.abs(c_f_k @ G_removed) ** 2) + self.a_n[k] * (np.abs(signal_f) ** 2)
         # 计算近用户和远用户的 SINR：
         rho_n = x_n / (interference_n + (self.K - 1) * self.awgn_var)
         rho_f = x_f / (interference_f + (self.K - 1) * self.awgn_var)


         # 防止对数计算无效
         eps = 1e-10  # 很小的数，防止除零或对零取对数
         if x_n > 0 and x_f > 0:
            rho_n = x_n / (interference_n + (self.K - 1) * self.awgn_var)
            rho_f = x_f / (interference_f + (self.K - 1) * self.awgn_var)
            reward += np.log2(1 + max(rho_n, eps)) + np.log2(1 + max(rho_f, eps))
         else:
            reward += 0  # 如果x_n或者x_f无效，设置默认奖励
         opt_reward += np.log2(1 + max(x_n, eps) / ((self.K - 1) * self.awgn_var)) + np.log2(1 + max(x_f, eps) / ((self.K - 1) * self.awgn_var))


        return reward, opt_reward

    def step(self, action):
        self.episode_t += 1

        action = action.reshape(1, -1)

       # 将动作向量切分为 G、a_n、a_f、Phi 部分
        G_real = action[:, :self.M * self.K]
        G_imag = action[:, self.M * self.K : 2 * self.M * self.K]
        a_n_raw = action[:, 2 * self.M * self.K : 2 * self.M * self.K + self.K]
        a_f_raw = action[:, 2 * self.M * self.K + self.K : 2 * self.M * self.K + 2 * self.K]
        Phi_real = action[:, -2 * self.L:-self.L]
        Phi_imag = action[:, -self.L:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
        # 标准化每对用户的功率分配系数
        a_n_raw = a_n_raw.flatten()
        a_f_raw = a_f_raw.flatten()
        self.a_n = np.zeros(self.K)
        self.a_f = np.zeros(self.K)
        for i in range(self.K):
           x = (a_n_raw[i] + 1.0) / 2.0
           y = (a_f_raw[i] + 1.0) / 2.0
           s = x + y
           if s < 1e-6:
              self.a_n[i] = 0.5
              self.a_f[i] = 0.5
           else:
              self.a_n[i] = x / s
              self.a_f[i] = y / s
         # 更新 RIS 相位
        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)
        magnitudes = np.abs(np.diag(self.Phi))          # 取对角元素模长
        self.Phi = np.diag(np.diag(self.Phi) / magnitudes)
        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2



        # 直接计算 power_r，而不再使用 H_2_tilde
        power_r = np.linalg.norm(self.H_rf + self.H_rn, axis=0).reshape(1, -1) ** 2


        H1_real = np.real(self.H_1).reshape(1, -1); H1_imag = np.imag(self.H_1).reshape(1, -1)
        H_bn_real = np.real(self.H_bn).reshape(1, -1); H_bn_imag = np.imag(self.H_bn).reshape(1, -1)
        H_bf_real = np.real(self.H_bf).reshape(1, -1); H_bf_imag = np.imag(self.H_bf).reshape(1, -1)
        H_rn_real = np.real(self.H_rn).reshape(1, -1); H_rn_imag = np.imag(self.H_rn).reshape(1, -1)
        H_rf_real = np.real(self.H_rf).reshape(1, -1); H_rf_imag = np.imag(self.H_rf).reshape(1, -1)
        self.state = np.hstack((action, power_t, power_r,
                             H1_real, H1_imag,
                             H_bn_real, H_bn_imag,
                             H_bf_real, H_bf_imag,
                             H_rn_real, H_rn_imag,
                             H_rf_real, H_rf_imag))

        reward, opt_reward = self._compute_reward(self.Phi)

        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass