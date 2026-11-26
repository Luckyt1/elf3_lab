# 11.26 V1.0 37000

这里保存了 2026 年 11 月 25 日 23:07 启动的固定水平速度扰动训练。项目根目录的训练源码和机器人资产来自该次启动时的原始快照，检查点为 `model_37000.pt`。承重脚相对参考落点的位移尺度为 **6 cm**，默认两脚内缘净距为 **8 cm**。

## 文件位置

| 内容 | 位置 |
|---|---|
| 第 37,000 轮模型 | [训练检查点](model/model_37000.pt)（含 Actor、Critic 和 Adam） · [ONNX 策略](model/policy.onnx) · [TorchScript 策略](model/policy.pt) · [输入输出与关节顺序](model/policy_interface.json) |
| 原分支训练曲线 | [曲线图](curves/training_0_to_37000.png) · [逐轮 CSV](curves/training_0_to_37000.csv) · [TensorBoard 数据](curves/tensorboard/) |
| 第 37,000 轮摆臂回放曲线 | [左右脚与骨盆的水平位移](curves/replay_37000.png) · [逐帧 CSV](curves/replay_37000.csv) |
| 训练入口 | [启动脚本](scripts/gl165.sh) · [训练主程序](legged_lab/scripts/train.py) |
| 站立任务 | [站姿、奖励权重和 PPO 配置](legged_lab/envs/gl_165/stand_cfg.py) · [环境与恢复逻辑](legged_lab/envs/gl_165/stand_env.py) · [奖励公式](legged_lab/envs/gl_165/rewards.py) |
| PPO 与网络实现 | [训练器](rsl_rl/rsl_rl/runners/on_policy_runner.py) · [PPO](rsl_rl/rsl_rl/algorithms/ppo.py) · [Actor 和 Critic](rsl_rl/rsl_rl/modules/actor_critic.py) |
| 机器人与手臂动作 | [训练用 URDF、MuJoCo XML 与网格](legged_lab/assets/gl_165/urdf_261123_2_5ms_stance8cm/) · [128 段手臂动作库](legged_lab/envs/gl_165/datasets/) |
| 检查点和运行快照 | [原训练目录](logs/stand_gl165/2026-11-25_23-07-34_stand_540_xmz_stance8_torso_sway_fixed_xy_push_from_zero_4096_80000/) 里的 `model_37000.pt` 和 `params/`。后者保存了 `env.yaml`、`agent.yaml`、控制参数、动作库、物理资产、原始源码 ZIP 和来源校验值。 |
| 运行记录 | [原始启动命令](records/launch.json) · [该检查点的无推摆臂回放](records/in_place_arms.json) |

## 模型与曲线

两个推理文件均来自同一个第 37,000 轮 Actor。输入为 `float32[1, 540]`（最近 10 帧，从旧到新），输出为 15 个原始动作。每个动作对应的关节、缩放及默认角记录在接口文件中。[导出校验](model/export_validation.json)

训练图显示平均回合奖励、平均回合时长、两项 PPO 损失和两项静站指标。浅色线是逐轮值，深色线是截至该轮最近 200 轮的均值。CSV 和 TensorBoard 保留以 PPO 轮数为横轴的全部 66 项原始指标，从第 0 轮到第 37,000 轮，共 37,001 行数据。CSV 沿用训练日志单位，图中将回合长度从控制步换算为秒，将承重脚速度换算为 cm/s。

摆臂回放图记录该模型在零速度指令、无外推条件下连续做完 128 段动作及随后站立 20 秒的过程。每只脚和骨盆分别以摆臂前 1 秒的水平位置中位数为参考。CSV 以第一个摆臂采样点为 0 秒，每 20 ms 一行。`phase=1` 表示手臂运动，`phase=2` 表示停臂后的站立。

在本版本目录可用 TensorBoard 交互查看训练指标。四个事件文件按轮次连续保存同一次训练，可直接打开目录。

```bash
tensorboard --logdir curves/tensorboard
```

## 训练设置

- 从第 0 轮初始化策略、价值网络和 Adam，随机种子为 42。4096 个环境中，512 个固定手臂静站，512 个无推摆臂，3072 个摆臂并接受水平速度扰动。
- 可受推的环境各自每隔 6–12 秒在世界系 X、Y 速度上分别叠加 −0.8 至 +0.8 m/s 的随机增量。受推后允许迈步恢复。
- 物理步长 5 ms，控制周期 20 ms。Actor 输入为 54 维 × 10 帧，输出为 15 维腰腿动作。Critic 输入为 73 维 × 5 帧。
- 此检查点记录的轮号为 37,000，Adam 步数为 740,020。9 组零指令、无外推摆臂回放均完成，未检测到自发迈步或跌倒。计入停臂后的站立，左右脚相对动作前落点的最大水平位移分别为 0.86 cm 和 1.65 cm。

## 核对与使用

从本版本目录执行校验。成功时命令安静退出。

```bash
sha256sum --quiet --check SHA256SUMS
```

训练入口使用 Isaac Lab 2.1、Isaac Sim 4.5 和 PyTorch。先激活配置好这些依赖的 Python 环境，再从本版本目录设置解释器和运行名称。原训练的非交互启动在已接受 NVIDIA Omniverse 许可的环境中设置了 `OMNI_KIT_ACCEPT_EULA=YES`。

```bash
export GL165_PYTHON="$(command -v python3)"
run=2026-11-25_23-07-34_stand_540_xmz_stance8_torso_sway_fixed_xy_push_from_zero_4096_80000
```

需要从第 37,001 轮继续训练时，以下命令会载入检查点的 Actor、Critic 和 Adam，沿用 6 cm 位移尺度，累计目标为 80,000 轮。新日志写入本目录下的 `logs/stand_gl165/`。

```bash
bash scripts/gl165.sh train --headless --num_envs 4096 --seed 42 \
  --resume True --load_run "$run" --checkpoint model_37000.pt \
  --total_iterations 80000 --run_name stand_6cm_resume37000
```

从第 0 轮按原配置重新训练，可使用当时的启动参数。

```bash
bash scripts/gl165.sh train --headless --num_envs 4096 --seed 42 \
  --xmz-standing --max_iterations 80000 \
  --run_name stand_540_xmz_stance8_torso_sway_fixed_xy_push_from_zero_4096_80000
```
