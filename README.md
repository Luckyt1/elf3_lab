# GL-165 站立训练 · 11.26 V1.0 37000

仓库根目录是训练出 `model_37000.pt` 的代码。它来自 2026 年 11 月 25 日 23:07 开训时保存的快照。这次训练使用 8 cm 默认站姿、6 cm 静站承重脚位移尺度和固定水平速度扰动。Actor 每 20 ms 读取 54 维观测，堆叠 10 帧后输出 15 维腰腿动作。

| 内容 | 文件 |
| --- | --- |
| 训练入口和 PPO | [启动脚本](scripts/gl165.sh) · [训练主程序](legged_lab/scripts/train.py) · [训练循环](rsl_rl/rsl_rl/runners/on_policy_runner.py) · [PPO](rsl_rl/rsl_rl/algorithms/ppo.py) |
| 训练任务 | [站姿、奖励权重、推扰和手臂配置](legged_lab/envs/gl_165/stand_cfg.py) · [站立环境](legged_lab/envs/gl_165/stand_env.py) · [奖励公式](legged_lab/envs/gl_165/rewards.py) |
| 训练资产 | [URDF、MuJoCo 模型和网格](legged_lab/assets/gl_165/urdf_261123_2_5ms_stance8cm/) · [训练用手臂动作库](legged_lab/envs/gl_165/datasets/arm_disturbance_overhead_261123_2_5ms_stance8cm.npz) |
| 第 37,000 轮权重 | [含 Actor、Critic 和 Adam 的检查点](versions/11.26-V1.0-37000/model/model_37000.pt) · [ONNX](versions/11.26-V1.0-37000/model/policy.onnx) · [TorchScript](versions/11.26-V1.0-37000/model/policy.pt) · [模型接口](versions/11.26-V1.0-37000/model/policy_interface.json) |
| 训练和回放曲线 | [TensorBoard 事件日志](versions/11.26-V1.0-37000/curves/tensorboard/) · [训练曲线](versions/11.26-V1.0-37000/curves/training_0_to_37000.png) · [逐轮 CSV](versions/11.26-V1.0-37000/curves/training_0_to_37000.csv) · [无推摆臂回放](versions/11.26-V1.0-37000/curves/replay_37000.png) |
| 原始运行记录 | [开训命令](versions/11.26-V1.0-37000/records/launch.json) · [开训时的配置、源码 ZIP 和校验值](versions/11.26-V1.0-37000/logs/stand_gl165/2026-11-25_23-07-34_stand_540_xmz_stance8_torso_sway_fixed_xy_push_from_zero_4096_80000/params/) |

## 从根目录训练

在 Linux 上激活已配置 Isaac Sim 4.5、Isaac Lab 2.1、Isaac Lab Tasks 和 PyTorch 的 Python 3.10 环境，进入仓库根目录。启动脚本会优先加载本仓库的 `legged_lab` 和 `rsl_rl`。

```bash
export GL165_PYTHON="$(command -v python3)"
```

按原配置从第 0 轮开始训练。4096 个环境中，512 个固定手臂静站，512 个无推摆臂，3072 个摆臂并每隔 6–12 秒接受一次水平速度扰动。世界系 X、Y 各自均匀抽取 −0.8 至 +0.8 m/s，从开训起采用完整范围。

```bash
./scripts/gl165.sh train --headless --num_envs 4096 --seed 42 \
  --xmz-standing --max_iterations 80000 \
  --run_name stand_540_xmz_stance8_torso_sway_fixed_xy_push_from_zero_4096_80000
```

从第 37,001 轮继续训练时，下面的命令会读取已归档的模型、优化器和控制配置，目标累计为 80,000 轮。根目录 `logs/stand_gl165/` 中的原始运行名称链接到版本目录里的检查点，新日志写入独立的运行目录。

```bash
run=2026-11-25_23-07-34_stand_540_xmz_stance8_torso_sway_fixed_xy_push_from_zero_4096_80000
./scripts/gl165.sh train --headless --num_envs 4096 --seed 42 \
  --resume True --load_run "$run" --checkpoint model_37000.pt \
  --total_iterations 80000 --run_name stand_6cm_resume37000
```

## 查看曲线

下面的命令可读取原训练从 0 到 37,000 轮的全部 66 项指标。

```bash
tensorboard --logdir versions/11.26-V1.0-37000/curves/tensorboard
```

[版本归档和回放结果](versions/11.26-V1.0-37000/README.md) · [11 月 23 日版本及对应的使用说明](https://github.com/Luckyt1/elf3_lab/tree/9ed9218e2409b3daa5847181a49ca06e8a2cf70c) · [RSL-RL 许可](rsl_rl/LICENSE)
