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
annotate_videos.py
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

# 5. 按 source 核对类别构成与场景覆盖
python prepare_data.py stats

# 6. 划分方案确定后才生成实验 manifest
python prepare_data.py manifest
```

可以用 `--sources data_lm data_sr data_vj` 只处理部分 source。`all` 默认运行
inventory、extract、assemble、audit，但不会自动生成 stats 和 manifest。

`stats` 只读 `data/processed` 下已装配的序列，所以统计的是 15 FPS 时间轴上、过完
质量流程之后的构成，与 `data/inventory/summary.json`（原始标注、源 FPS、含下游会
丢弃的样本）刻意区分。不带 `--sources` 时每个 source 一份报告；带 `--sources` 时
把列出的 source **合并**成一份，例如 `--sources data_lm data_sr` 给出这两个文件夹
合起来的构成。每份报告列出按类的动作段数与占比、缺失的类别、以及覆盖的场景——场景
按 `jog (positive)` 的形式标注极性并按极性分组，便于核对负向场景的覆盖。结果同时
落到 `data/stats/dataset_stats.json` 和 `data/stats/class_stats.csv`。

真实路径依赖 `opencv-python` 和 `mediapipe`。昂贵的 MediaPipe 原始轨迹保存在
`data/cache/tracks/`；平滑、目标 FPS、palm 归一化、标签对齐和质量阈值在后续阶段，
修改这些参数不需要重新运行 MediaPipe。

视频不是镜像输入。根据 MediaPipe 的自拍镜像约定，物理右手对应输出标签 `Left`。
程序只接受该候选，低置信度或只检测到物理左手时记录 `v_mask=0`，绝不回退到左手。

## 给新视频打标注（annotate_videos.py）

拿到一批新录的视频，想直接看模型预测什么、准不准，用这条链路。它从 mp4 一路跑到
标注文件，不需要事先有标注，也不需要 manifest：

```bash
python annotate_videos.py --config config/config_annotate.yaml --video-root ~/new_videos

# 先拿两条试跑，确认 MediaPipe 能抽到手
python annotate_videos.py --video-root ~/new_videos --limit 2

# 额外产出带软字幕的 mp4，播放时字幕与画面同步
python annotate_videos.py --video-root ~/new_videos --mux
```

内部依次是 扫描 → MediaPipe 抽轨迹 → 装配 128 维特征 → 重叠窗口推理 → 写产物，
复用 `prepare_data.py` 的同一套缓存，所以中断后重跑会跳过已完成的视频。
`config/config_annotate.yaml` 里的 `extract` 和 `assemble` 两段必须与训练时一致，
否则 checkpoint 的 fingerprint 校验会直接报错——这是有意的保护，不要绕过。

### 目录布局与命名

**与视频命名无关，一个 mp4 就是一个样本。** 两种布局可以混在同一个根目录下：

```text
new_videos/
├── DJI_20260613115350_LN043-bike1-5.mp4     # 扁平：无标签测试数据
├── VID_20260613_090940_LN044-stand4-5.mp4
└── DG2026091502_F_R_walk/                   # 训练布局：文件夹内单个视频
    ├── recording.mp4
    └── NOVA project/gestures.annotation~
```

判定规则只有一条：**某个文件夹里只有一个视频时，该文件夹就是样本目录**（这是训练
数据的布局，也只有这种情况才会按 `<subject>_<gender>_<field3>_<scene>` 解析目录名）；
其余情况给每个视频建一个以文件名命名的子目录。所以上面的例子会产出：

```text
new_videos/
├── DJI_20260613115350_LN043-bike1-5/
│   ├── NOVA project/gestures.annotation~    # 预测标注
│   ├── gestures_timeline.png                # 时间轴条带图
│   ├── gestures.srt
│   └── ..._pred.mp4                         # 仅 --mux
└── DG2026091502_F_R_walk/
    ├── NOVA project/gestures.annotation~    # 人工真值，不会被动
    ├── NOVA project/gestures.pred.annotation~
    └── gestures_timeline.png                # 预测 / 真值 / 错误 三行
```

### 怎么看结果

先看 `gestures_timeline.png`：横轴是时间，颜色代表类别，配色在所有视频间固定，
所以不同视频可以横着对比。有真值时会多出真值行和标红的错误行，标题里带 frame
accuracy，一整段视频对错一眼扫完，不用播放。只有需要逐秒核对时才去看软字幕版
mp4（`--mux`），或者用播放器手动挂载 `gestures.srt`。

标注文件每行是 `起始秒;结束秒;类别id;置信度;`，与训练数据同格式，可以直接用 NOVA
打开修正后当训练数据用。置信度是该片段内预测类别的 softmax 均值，按它排序能快速
挑出最该人工复核的片段。

### 两条安全机制

- **不覆盖人工标注**：目标位置已有 `gestures.annotation~` 时，预测写到同目录的
  `gestures.pred.annotation~` 并打 warning。确实想覆盖时传 `--overwrite-annotation`。
- **不把自己的输出当成真值**：脚本写出的标注旁边会留一个 `.gestures.predicted.json`
  标记。重跑时凭它识别出"这是上次预测的"，原地覆盖而不是堆一堆 `.pred.` 文件，也
  不会拿上次的预测当真值去算准确率。`--mux` 产出的 `*_pred.mp4` 同样会被扫描跳过。

无标签序列写到 `data/processed_infer/`，刻意与训练语料 `data/processed/` 分开，
避免被 `prepare_data.py manifest` 扫进训练集。

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
- 推理：每个视频的 `frames.csv`、`prediction.npy`、`segments.json`。序列自带标签时
  `frames.csv` 会多出 `true_label` / `true_name` / `correct` 三列，筛 `correct=0`
  就能直接定位错帧
- 评估：`outputs/evaluation_results/.../metrics.json`，同目录还有
  `per_class_metrics.csv`（每类的帧级与段级 P/R/F1、样本量、最常被错判成哪一类）和
  `confusion_matrix.csv`。日志会按帧级 F1 从差到好打印类别表，并单独提示测试集中
  完全缺失的类别
- 标注：`outputs/annotate_results/.../annotations.json` 索引，以及每个视频的
  `prediction.npy` / `logits.npy`，事后复查不用重跑

Checkpoint 自带模型名、参数、类别映射、manifest hash、视频列表、整视频验证指标和
预处理 schema/fingerprint。输入数据和训练配方不兼容时，推理会明确报错。

## 测试

```bash
python -m unittest discover -s tests -v
```

单元测试覆盖标注解析与写回、稳健聚合、缓存失效、用户级划分、动态窗口、窗口掩码、
模型形状、损失、重叠拼接、指标，以及标注链路的目录扫描（扁平与训练两种布局）和
覆盖保护。MediaPipe 检出质量与真实时间对齐只有在拿到可运行的加密数据后才能验证。

标注时间戳按 0.1ms 量化并向内收一格再写出。`labels_from_intervals` 对结束时间取
`ceil`，边界值经浮点乘法会溢出到下一帧（`2.24 * 25` 在 IEEE 下是
`56.00000000000001`），不收这一格的话每个预测片段都会被系统性地拉长一帧。
`test_round_trip_survives_timestamp_rounding` 用 5 种帧率守住这个行为。
