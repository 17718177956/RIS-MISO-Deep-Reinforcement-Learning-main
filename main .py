import argparse
import os

import numpy as np
import torch

import DDPG
import utils

import environment
import matplotlib.pyplot as plt




def whiten(state):
    return (state - np.mean(state)) / np.std(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choose the type of the experiment
    parser.add_argument('--experiment_type', default='custom', choices=['custom', 'power', 'rsi_elements', 'learning_rate', 'decay'],
                        help='Choose one of the experiment types to reproduce the learning curves given in the paper')

    # Training-specific parameters
    parser.add_argument("--policy", default="DDPG", help='Algorithm (default: DDPG)')
    parser.add_argument("--env", default="RIS_MISO", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=0, type=int, metavar='N', help='Number of exploration time steps sampling random actions (default: 0)')
    parser.add_argument("--buffer_size", default=1000000, type=int, help='Size of the experience replay buffer (default: 100000)')
    parser.add_argument("--batch_size", default=64, metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    # Environment-specific parameters
    parser.add_argument("--num_antennas", default=4, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--num_RIS_elements", default=4, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=4, type=int, metavar='N', help='Number of users')
    parser.add_argument("--power_t", default=30, type=float, metavar='N', help='Transmission power for the constrained optimization in dB')
    parser.add_argument("--num_time_steps_per_eps", default=10000, type=int, metavar='N', help='Maximum number of steps per episode (default: 20000)')
    parser.add_argument("--num_eps", default=10, type=int, metavar='N', help='Maximum number of episodes (default: 5000)')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')

    # Algorithm-specific parameters
    parser.add_argument("--exploration_noise", default=5000, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.90, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=0.01, type=float, metavar='G',  help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    parser.add_argument("--actor_lr",  default=3e-4, type=float,help="Learning rate for the Actor network")
    parser.add_argument("--critic_lr", default=1e-3, type=float,help="Learning rate for the Critic network")
    parser.add_argument("--decay", default=1e-4, type=float, metavar='G', help='Decay rate for the networks (default: 0.00001)')
    parser.add_argument("--noise_clip", default=0.5, type=float, help='目标策略噪声截断范围（默认:0.5）')
    parser.add_argument("--policy_noise", default=0.2, type=float,help='目标策略平滑的高斯噪声标准差（默认:0.2）')
    parser.add_argument("--policy_freq", default=2, type=int, help='策略网络延迟更新频率 d（默认:2）')


    args = parser.parse_args()
    args.num_time_steps_per_eps = 5000    # ← 换成你想要的上限
    args.num_eps = 10   # 只跑一个 Episode

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    file_name = (f"{args.num_antennas}_{args.num_RIS_elements}_{args.num_users}_{args.power_t}_"f"A{args.actor_lr}_C{args.critic_lr}_{args.decay}")


    if not os.path.exists(f"./Learning Curves/{args.experiment_type}"):
        os.makedirs(f"./Learning Curves/{args.experiment_type}")

    if args.save_model and not os.path.exists("./Models"):
        os.makedirs("./Models")

    env = environment.RIS_MISO(args.num_antennas, args.num_RIS_elements, args.num_users, AWGN_var=args.awgn_var)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "power_t": args.power_t,
        "max_action": max_action,
        "M": args.num_antennas,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "actor_lr":  args.actor_lr,
        "critic_lr": args.critic_lr,
        "actor_decay": args.decay,
        "critic_decay": args.decay,
        "device": device,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq
    }

    # Initialize the algorithm
    agent = DDPG.DDPG(**kwargs)
        # -------- 新增：经验回放池 --------
    replay_buffer = utils.ExperienceReplayBuffer(
        state_dim      = state_dim,
        action_dim     = action_dim,
        max_size       = args.buffer_size
    )

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")


    # Initialize the instant rewards recording array
    instant_rewards = []
    near_trend1, far_trend1 = [], []
    max_reward = 0

    instant_rewards = []                    # list to record per-episode reward sequences
    near_trend1, far_trend1 = [], []        # (unchanged: for power allocation trend)
    max_reward = 0
     # **Initialize interactive plot for sum-rate trend**
    plt.ion()                               # turn on interactive mode
    fig, ax = plt.subplots()                # create figure for live plotting
    global_step = 0
    x_vals = []                             # global step indices for plot
    sum_rate_vals = []                      # cumulative sum-rate values for plot

for eps in range(int(args.num_eps)):
    episode_reward = 0.0
    state_raw = env.reset()                  # 环境原始状态
    state_net = whiten(state_raw)            # 给神经网络用

    eps_rewards = []                   # list of rewards for this episode

    for t in range(int(args.num_time_steps_per_eps)):
        # Choose action according to current policy
        action = agent.select_action(state_net)


        # -- 调试输出（每 100 步打印一次）--
        if t % 100 == 0:            # 改成想要的频率
          a_tensor = torch.from_numpy(action).float().to(device)
          p_out = agent.actor.compute_power(a_tensor).item()
          limit = np.sqrt(10 ** (args.power_t / 10))

         # 从 action 切片算 a_n / a_f
          start = 2 * args.num_antennas * args.num_users
          end   = start + 2 * args.num_users
          a_raw = action[0, start:end]
          x = (a_raw[:args.num_users]         + 1) / 2.0   # 归一化到正区间
          y = (a_raw[args.num_users:2*args.num_users] + 1) / 2.0
          s = x + y + 1e-12
          a_n = (x / s)[0]                      # 只看第 1 对用户
          a_f = (y / s)[0]

          print(f"[t={t}] ||w|| = {p_out:.3f}/{limit:.3f}  "
          f"a_n+a_f = {a_n+a_f:.3f}  (a_n={a_n:.3f}, a_f={a_f:.3f})")
                # -----------------------------------

        # Log first user-pair power allocation (for analysis plot)
        a_start = 2 * args.num_antennas * args.num_users
        a_end   = a_start + 2 * args.num_users
        a_vals = action[0][a_start:a_end]
        near_trend1.append(a_vals[0])             # near user power coeff
        far_trend1.append(a_vals[args.num_users]) # far user power coeff

        # Take the action in the environment
        next_state_raw, raw_reward, done_signal, _ = env.step(action)
        next_state_net = whiten(next_state_raw)
        scaled_reward = raw_reward       # ⭐ 仅这一行做缩放
        if t % 10000 == 0:
           phi_mag = np.abs(np.diag(env.Phi))
           print(f"[t={t}] Φ|·|均值={phi_mag.mean():.3f},  最小={phi_mag.min():.3f}, 最大={phi_mag.max():.3f}")
           # ---- 计算并打印瞬时总速率（单位：bps/Hz）----
        sum_rate = float(np.sum(raw_reward))     # reward 已经是每个用户速率之和
        if t % 100 == 0:                     # 每 100 步打印一次，自己调频率
         print(f"[t={t}]  Sum-rate = {sum_rate:.3f} bps/Hz")
            # ----------------------------------------------
        phi_angle = np.angle(np.diag(env.Phi))          # [-π, π]
        if t % 10000 == 0:
         print(f"[t={t}]  Φ∠均值={phi_angle.mean():+.3f} rad,  "
            f"最小={phi_angle.min():+.3f}, 最大={phi_angle.max():+.3f}")

        # 3. 现在才能检查 a_n + a_f = 1
        with torch.no_grad():
            assert np.allclose(env.a_n + env.a_f, 1, atol=1e-3), "a_n + a_f ≠ 1"

        # Determine if episode is done (True if env signaled done or time limit reached)
        done_flag = bool(done_signal) or (t == args.num_time_steps_per_eps - 1)
        done = 1.0 if done_flag else 0.0  # store as float (1.0 or 0.0) for replay buffer

        # Store transition in replay buffer
        replay_buffer.add(state_net, action, next_state_net, scaled_reward, done)

        # Update state and cumulative reward
        episode_reward += float(np.sum(raw_reward))   # accumulate reward (ensure it's a scalar)
        state_raw  = next_state_raw
        state_net  = next_state_net

        # Update the agent (DDPG) using a batch from replay buffer
        agent.update_parameters(replay_buffer, args.batch_size)

        # Log reward for this time step (for analysis and plotting)
        reward_scalar = float(np.sum(scaled_reward))
        eps_rewards.append(reward_scalar)
        if t % 500 == 0:  
         # Print progress (time step, episode, reward)
         print(f"Time step: {t+1}, Episode: {eps+1}, Reward: {reward_scalar:.3f}")


        if done_flag:
            # Episode ends here
            print(f"\nEnd of Episode {eps+1}: Steps = {t+1}, Episode Reward = {episode_reward:.3f}, Max Reward so far = { (max_reward[0].item() if isinstance(max_reward, np.ndarray) else float(max_reward)):.3f}\n")
            instant_rewards.append(eps_rewards)  # save this episode's reward sequence
            # Save rewards to file for this episode
            np.save(f"./Learning Curves/{args.experiment_type}/{file_name}_episode_{eps+1}", instant_rewards)
            break  # break out of the time-step loop to start a new episode
