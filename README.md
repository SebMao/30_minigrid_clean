# 30_minigrid_clean 项目结构与多智能体搜索算法说明

本项目基于 MiniGrid 与 MarlGrid，围绕多智能体搜索与强化学习训练展开。`main-mappo.py` 是训练骨干代码，负责环境构建、采样、优势/回报计算以及策略/价值网络的更新与日志记录。

## 目录结构

- `main-mappo.py`：多智能体 PPO 训练主程序（骨干）
- `main-ppo.py`：单智能体/简化 PPO 训练示例（其它智能体动作固定）
- `main.py`：环境演示与手动驱动示例
- `environment.py`：环境包装器，负责观测转换、渲染与交互

## 运行环境与依赖

- 主要依赖：`torch`、`numpy`、`gymnasium`、`minigrid`、`marlgrid`、`pygame`、`tensorboard`、`tqdm`
- 渲染：基于 `pygame`；日志：`tensorboard` 写入到 `runs/` 目录

## 算法主线（以 main-mappo.py 为主）

### 参数与超参

- 解析入口：`main-mappo.py:28`（`parse_args`）
- 关键超参：`num_steps`（默认 200）、`total_timestamps`、`learning_rate`、`gamma`、`gae` 与 `gae_lambda`、`clip_coef`、`vf_coef`、`update_epochs`、`num_minibatches`
- 批量规模：`batch_size = num_envs * num_steps`，`minibatch_size = batch_size // num_minibatches`（`main-mappo.py:82-84`）

### 环境构建

- 注册多智能体环境：`register_marl_env("MyEnv", ClutteredMultiGrid, n_agents=3, grid_size=19, view_size=5, ...)`（`main-mappo.py:181-188`）
- 创建并包裹：`env = gym.make("MyEnv")`，`env = RescueEnv(env)`（`main-mappo.py:189-193`）
- 包装器职责：
  - 全局观测 `get_s0`（`environment.py:56-64`）：将网格字符串转张量并归一化；当前返回 `ret * viewed_area * 0`（关闭全局地图）
  - 局部观测 `get_s_agent`（`environment.py:66-72`）：每个智能体的局部 5×5 视图
  - 步进与渲染 `step`（`environment.py:78-89`）：调用底层环境并渲染

### 模型结构

- 策略网络 `Actor`（`main-mappo.py:87-121`）：
  - 全局分支：3×Conv + `fc1`，输出 `121` 维（19×19 展平）
  - 局部分支：`conv_local (3×3)` + `fc2`，输出 `25` 维（5×5 展平）
  - 融合：拼接得到 `146` 维，经 `fc` 输出 `3` 维 logits（动作类别）
- 价值网络 `Critic`（`main-mappo.py:123-155`）：
  - 全局分支同上
  - 多智能体局部分支：对每个智能体的局部视图经 `conv_local (1→3)` 与 `fc2 (75→25)`，再拼接
  - 融合：`[121 + 25×agents] → fc → 标量值`（`main-mappo.py:136-155`）

### 缓存与采样

- 张量缓存：
  - `maps`：`[num_steps, map_size, map_size]`（`main-mappo.py:201`）
  - `local_obses`：`[num_steps, agents, view_size, view_size]`（`main-mappo.py:202`）
  - `actions`、`logprobs`、`rewards`、`dones`、`values`（`main-mappo.py:204-209`）
- 采样循环：
  - 对每步记录当前观测与 done（`main-mappo.py:232-237`）
  - 对每个智能体：`actor(map, local)` 采样动作与对数概率（`main-mappo.py:241-246`）
  - 评估当前值：`critic(map, local_all_agents)`（`main-mappo.py:246-247`）
  - 环境步进：`env.step(actions[step])`（`main-mappo.py:250`）并记录奖励与下一个观测（`main-mappo.py:251-259`）

### 优势与回报

- GAE 分支：`advantages[t] = δ_t + γλ * ...`（`main-mappo.py:264-276`），`returns = advantages + values.unsqueeze(1)`（`main-mappo.py:276`）
- 非 GAE：反向累积回报（`main-mappo.py:278-289`），`advantages = returns - values`（`main-mappo.py:289`）
- 聚合回报：`b_returns = returns.sum(1)`（按步聚合多智能体奖励；`main-mappo.py:297`）

### 更新与优化

- 比率计算：`ratio = exp(newlogprob - oldlogprob)`（`main-mappo.py:313-315`）
- 策略损失：`max(pg_loss1, pg_loss2)`，其中 `pg_loss1 = -adv * ratio`，`pg_loss2 = -adv * clamp(ratio, 1±clip)`（`main-mappo.py:334-339`）
- 价值损失：`MSE(newvalue, b_returns)`，可选值函数裁剪（`main-mappo.py:340-354`）
- 总损失与优化：`loss = pg_loss + v_loss * vf_coef`，优化 `Actor` 与 `Critic`（`main-mappo.py:356-365`）
- KL 与裁剪统计：`old_approx_kl`、`approx_kl`、`clipfrac`（`main-mappo.py:316-321`）

### 日志与度量

- `SummaryWriter` 初始化与超参记录（`main-mappo.py:167-171`）
- 写入 `learning_rate`、`value_loss`、`policy_loss`、`returns`、`KL`、`clipfrac`、`explained_variance`、`SPS`（`main-mappo.py:375-388`）

## 关键模块与文件说明

### environment.py（环境包装器）

- 观测编码：将字符网格编码为数值并归一化（`environment.py:34-54`）
- 全局状态：当前返回全零张量以关闭全局地图影响（`environment.py:63`）
- 局部视图：按智能体生成 5×5 局部观测（`environment.py:66-72`）
- 步进/渲染：调用底层 MarlGrid 并在每步渲染（`environment.py:78-89`）

### main-ppo.py（单智能体训练示例）

- 仅训练一个智能体，其余智能体动作为固定常量（`main-ppo.py:262`）
- `Actor` 结构更简洁，直接基于局部观测产生 3 类动作（`main-ppo.py:87-134`）
- 优势默认使用非 GAE，`advantages = returns`（`main-ppo.py:301-306`）

### main.py（演示脚本）

- 注册并构建环境，包装为 `RescueEnv`，启动渲染与简单步进（`main.py:8-24`）

## 状态与动作设计

- 全局状态 `s0`：形状约 `[1, 1, 19, 19]`，当前被置零以进行仅局部观测实验（`environment.py:63`）
- 局部状态 `si`：每个智能体 `[1, 1, 5, 5]`
- 动作空间：`Actor` 输出 3 类动作；与 MiniGrid 的完整动作集合不同，本项目将动作约束到 3 类（例如前进/左转/右转——具体映射由底层环境动作空间与包装器约定）

## 运行方式

- 多智能体训练：
  ```bash
  python main-mappo.py --total-timesteps 100000 --num-steps 200 --learning-rate 5e-4
  python main-mappo-refactor.py --total-timesteps 100000 --max-episode-steps 200 --learning-rate 5e-4
  ```
- 单智能体训练示例：
  ```bash
  python main-ppo.py --total-timesteps 100000 --num-steps 2000
  ```
- 环境演示：
  ```bash
  python main.py
  ```
- 模型评估：
  ```bash
  python eval_mappo.py --model-path runs/YOUR_RUN_NAME/actor.pth  ## 评估模型性能
  python eval_mappo.py --model-path runs/YOUR_RUN_NAME/actor.pth --render --num-episodes 5  ## 环境渲染```
  ```
## 设计注意与扩展建议

- 学习率退火：按更新轮次线性退火（`main-mappo.py:223-227`）
- GAE/非 GAE 可切换（`main-mappo.py:264-290`）
- 回报聚合：多智能体奖励按步求和以更新共享价值（`main-mappo.py:297`）
- 渲染频率：包装器每步渲染，训练时可能影响速度（`environment.py:87-89`）
- 环境规模与智能体数：通过 `register_marl_env` 的参数更改（`main-mappo.py:181-188`）
- 模型结构：可根据任务需要扩展 `Actor` 的输出维度与 `Critic` 的聚合方式

## 日志与可视化

- TensorBoard：日志在 `runs/` 目录下，运行 `tensorboard --logdir runs` 可视化训练过程
- 关键指标：`returns`（批次总奖励）、`value_loss`、`policy_loss`、`approx_kl`、`clipfrac`、`SPS`（每秒步数）
