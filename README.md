# GL-165 站立抗扰动训练

这个项目训练 GL-165 在手臂做动作、躯干受到水平推力时维持站立。策略控制腰腿的 15 个关节，双臂的 14 个关节按动作库运动。受推时允许迈步，重新站稳后以新位置为参考。两脚足底内缘的横向净距有 3 cm 软下限。

## 运行环境

训练和评估使用 Linux、Python 3.10、Isaac Sim 4.5、Isaac Lab v2.1.0 和 PyTorch 2.5.1。先激活配好 Isaac Lab 和 Isaac Lab Tasks 的环境，再在本仓库根目录运行以下命令。启动脚本使用当前环境的 `python3`，也可以通过 `GL165_PYTHON` 指定解释器。

```bash
python3 -m pip install GitPython tensorboard
```

MuJoCo 回放可以使用独立的 CPU 环境。本项目使用 MuJoCo 3.17.0 验证了回放。

```bash
python3 -m pip install torch==2.5.1 'numpy<2' GitPython mujoco==3.17.0
```

## 训练

先用小规模训练检查环境。首次启动时，Isaac Lab 会把仓库内的 URDF 转换为 USD。

```bash
./scripts/gl165.sh train --headless --num_envs 16 --total_iterations 2 --run_name smoke
```

正式训练从零开始，使用 4096 个环境和 80000 轮。每 100 轮保存一次检查点，训练曲线、配置、动作库和代码快照位于同一运行目录。

```bash
./scripts/gl165.sh train --headless --num_envs 4096 --total_iterations 80000 --run_name gl165_540
tensorboard --logdir logs/stand_gl165
```

检查点位于 `logs/stand_gl165/<运行目录>/model_<轮次>.pt`。例如，下列命令选出最近一次小规模训练产生的检查点。评估训练效果时，将 `checkpoint` 指向正式训练的模型。

```bash
checkpoint=$(find logs/stand_gl165 -path '*_smoke/model_1.pt' -print | sort | tail -n 1)
./scripts/gl165.sh evaluate --headless --num_envs 8 --steps 100 --checkpoint "$checkpoint"
./scripts/gl165.sh mujoco --checkpoint "$checkpoint" --duration 15 --push
```

Isaac 评估默认按训练设置抽取两类扰动，使用 `--no_push` 可以测试单纯的手臂动作。MuJoCo 使用检查点中的关节顺序、站姿、动作缩放、软限位和 PD 增益进行独立动力学回放。预览时手臂按完整幅度运动，`--push` 在第 2 秒沿世界系 +Y 方向推一次，力和时长取自检查点所在轮次。加 `--headless` 可以无窗口运行。

## 策略和扰动

| 项目 | 当前配置 |
| --- | --- |
| 策略输入 | 每帧依次为机身角速度 3、机身系重力方向 3、速度指令 3、腰腿相对站姿关节角 15、腰腿关节速度 15、上一次动作 15。关节速度乘 0.1，站立指令为零 |
| 历史与输出 | 每 20 ms 采样，按从旧到新的顺序堆叠 10 帧，共 540 维，输出 15 个腰腿动作 |
| 训练时的价值网络 | 每帧 73 维、5 帧历史。价值网络可使用接触和恢复信息，部署策略只使用上表的六类输入 |
| 物理和手臂 | 物理步长 5 ms，每个策略周期 4 个物理子步，每段手臂过渡 0.6 s |
| 训练扰动 | 每个环境独立间隔 8–12 s，两类扰动各占 50% |
| 瞬时速度扰动 | 满强度时世界系 X/Y 各轴最多 ±0.8 m/s，偏航最多 ±0.2 rad/s，前 4800 s 仿真时间由 5% 线性升至满强度 |
| 持续躯干推力 | 从 20 N、0.3 s 开始，到第 20000 轮线性升至 100 N、1.5 s，之后维持满强度 |

## 受推后的恢复判定

训练环境把已知推力或检测到的失稳作为恢复起点。恢复期间停止按原位置给分。推力解除后的 2 s 保护期结束，且以下条件连续满足 0.4 s，环境才把当时的骨盆水平位置设为新站立参考。两个计时可以重叠。

- 左右脚各承受至少 40 N 的竖直载荷。
- 骨盆水平速度不超过 0.12 m/s，倾角不超过 0.13 rad，横滚和俯仰的合成角速度不超过 0.45 rad/s。

这一判定用于训练评分。实际运行时，策略靠 540 维观测连续输出腰腿动作，无需部署端提供“恢复完成”信号。完整触发条件见[恢复判定代码](legged_lab/assets/gl_165/stance_recovery.py)。

## 代码与资产

- [站立任务配置](legged_lab/envs/gl_165/stand_cfg.py) 包含观测、奖励和扰动课程。
- [站立环境](legged_lab/envs/gl_165/stand_env.py) 实现动作映射、脚底约束和站稳后的参考位置更新。
- [机器人模型](legged_lab/assets/gl_165/) 包含训练用 URDF、MuJoCo 模型、几何网格与控制参数。[动作库](legged_lab/envs/gl_165/datasets/) 保存经过校验的手臂目标序列。发布时将来源路径改为相对路径，保留了对应旧检查点的摘要映射。
- [PPO 实现](rsl_rl/rsl_rl/) 随仓库提供。派生代码保留原作者声明，RSL-RL 的许可见 [LICENSE](rsl_rl/LICENSE)。

在安装了 PyTorch 和 NumPy 的环境中，可以运行模型、扰动、动作连续性与恢复判定的回归测试。

```bash
python3 -m unittest discover -s tests -v
```
