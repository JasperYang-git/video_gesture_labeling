# Video Gesture Labeling

离线摄像头手势时序分割 Pipeline。输入一段手势视频，模型输出逐帧动作类别（14 种手势 + background）。

当前仓库按配置驱动的方式重建了训练和推理流程。真实训练视频在另一台服务器上且无法访问，因此仓库内置确定性模拟数据，用来打通 `prepare → train → infer → evaluate`。MediaPipe 真实视频入口已经保留，拿到数据后只需关闭 mock 并指向原始目录。

## 任务与特征

- 任务：Temporal Action Segmentation，整段视频离线预测
- 模型：MS-TCN2（Prediction Generation + Refinement）
- 特征：128 维 `tracking_quality + v_mask + 63 坐标 + 63 速度`
- 评估：完整视频级 frame accuracy、Edit、F1@0.1/0.25/0.5

动作类别见 [`data/mapping.txt`](data/mapping.txt)。

## 目录

```text
train_script.py
infer_script.py
evaluate_script.py
prepare_data.py
config/
model/
utils/
data/mapping.txt
tests/
```

每段视频处理后保存为一个 `.npz`：

- `features[C, T]`
- `labels[T]`（可选）
- `valid_mask[T]`
- `video_id` / `fps` / schema 元数据

训练数据按动作片段动态生成窗口：每个 epoch 改变动作在窗口中的位置，默认不加入
纯背景窗口。验证和推理使用固定重叠窗口并融合为完整视频时间轴，最佳 checkpoint
按整视频 F1@0.5 选择。

## 模拟数据闭环

```bash
python -m pip install -r requirements.txt
python prepare_data.py mock --config config/config_prepare.yaml
python train_script.py --config config/config_smoke.yaml
python infer_script.py --config config/config_mock_infer.yaml
python evaluate_script.py --config config/config_mock_eval.yaml
```

模拟数据只用于通路验证，不要把模拟指标当成真实性能。

完整训练配置在 `config/config_train.yaml`。

## 大规模真实数据准备

服务器目录：

```text
~/video_dataset/
├── data_lm/
├── data_ln_once/
├── data_ln_twice/
├── data_sr/
└── data_vj/
```

每个样本目录命名为：

```text
<subject_id>_<gender>_<field3>_<scene>_<ignored...>
```

其中第一个字段是全局 subject，第 3 个字段不是左右手，第 4 个字段是完整场景名。
场景必须在 `config/scene_taxonomy.yaml` 中 exact 命中；未注册组合名（例如
`stand-alarm-strong`）标记为 unknown，默认不提取、不训练。

数据准备分为五个可独立重跑的阶段：

```bash
# 1. 只扫描目录、命名、标注和场景，不运行 MediaPipe
python prepare_data.py inventory

# 2. 每类抽一条，输出物理右手关键点叠图，必须人工检查
python prepare_data.py preview --limit 1

# 3. 每类先抽 20 条，4 workers 验证稳定性
python prepare_data.py extract --limit 20 --workers 4
python prepare_data.py assemble --limit 20
python prepare_data.py audit --limit 20

# 4. 稳定后全量；中断后执行同一命令会跳过有效缓存
python prepare_data.py extract --workers 7
python prepare_data.py assemble
python prepare_data.py audit

# 5. 划分方案确定后才生成实验 manifest
python prepare_data.py manifest
```

可以用 `--sources data_lm data_sr data_vj` 只处理部分 source。`all` 默认运行
inventory、extract、assemble、audit，但不会自动生成 manifest。

真实路径依赖 `opencv-python` 和 `mediapipe`。昂贵的 MediaPipe 原始轨迹保存在
`data/cache/tracks/`；平滑、目标 FPS、palm 归一化、标签对齐和质量阈值在后续阶段，
修改这些参数不需要重新运行 MediaPipe。

视频不是镜像输入。根据 MediaPipe 的自拍镜像约定，物理右手对应输出标签 `Left`。
程序只接受该候选，低置信度或只检测到物理左手时记录 `v_mask=0`，绝不回退到左手。

## Manifest、Fingerprint 与用户划分

- **Inventory**（`data/inventory/inventory.jsonl`）记录磁盘上发现的全部样本。
- **Manifest**（`data/manifests/*.json`）记录某次实验使用哪些 source/scene/polarity
  及其 train/validation/test。
- **Fingerprint** 是预处理配方与源数据的指纹。配置、视频大小/修改时间或标注内容
  变化时都会失效并自动重算，避免误用旧缓存。
- **Subject** 取完整第一个命名字段（例如 `LMVibra006`）。全局
  `data/manifests/subject_splits.json` 保证同一对象在五个 source 中不会跨 split。

正向视频只采含动作窗口，普通纯背景仍丢弃；taxonomy 中的负向视频必须是全
background，并作为受控 hard negative 采样，默认约占动作窗口数量的 15%。

## 输出

- 训练：`outputs/training_results/MMDD/MMDD_HHMMSS/`，最优权重另存为 `outputs/best_model.pth`
- 推理：每个视频的 `frames.csv`、`prediction.npy`、`segments.json`
- 评估：`outputs/evaluation_results/.../metrics.json`

Checkpoint 自带模型名、参数、类别映射、manifest hash、视频列表、整视频验证指标和
预处理 schema/fingerprint。输入数据和训练配方不兼容时，推理会明确报错。

## 测试

```bash
python -m unittest discover -s tests -v
```

单元测试覆盖标注解析、稳健聚合、缓存失效、用户级划分、动态窗口、窗口掩码、模型
形状、损失、重叠拼接和指标。MediaPipe 检出质量与真实时间对齐只有在拿到可运行的
加密数据后才能验证。
