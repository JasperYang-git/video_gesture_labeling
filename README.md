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
python prepare_data.py --config config/config_prepare.yaml --use-mock
python train_script.py --config config/config_smoke.yaml
python infer_script.py --config config/config_infer.yaml
python evaluate_script.py --config config/config_eval.yaml
```

`config/config_prepare.yaml` 默认 `mock.enabled: true`。这只用于通路验证，不要把模拟指标当成真实性能。

完整训练配置在 `config/config_train.yaml`。

## 真实视频

将数据放到 `data/users/default/<ID>_<Sex>_R_...>/`：

```text
video.mp4
NOVA project/gestures.annotation~
```

标注格式：

```text
time_begin;time_end;gesture_label;confidence;
```

时间为秒，`label=-1` 视为 background。然后：

```yaml
# config/config_prepare.yaml
mock:
  enabled: false
```

```bash
python prepare_data.py --config config/config_prepare.yaml
python train_script.py --config config/config_train.yaml
```

真实路径依赖 `opencv-python` 和 `mediapipe`。预处理顺序为：原始 FPS 提取与腕点
中心化、One-Euro 平滑、目标 FPS 时间桶聚合、palm 中位数归一化、速度计算。程序
同时生成 `data/processed/quality_audit_report.csv`，记录动作区间和分类别检测质量。

左右手按文件夹名第 3 段和 MediaPipe handedness 同时过滤，不再只取置信度最高的手。

## Manifest、Fingerprint 与用户划分

- **Manifest**（`data/processed/splits.json`）是数据目录，记录每个 NPZ 属于哪个 split，
  以及 `video_id`、`subject_id`、`session_id`、质量状态和预处理版本。
- **Fingerprint** 是预处理配方与源数据的指纹。配置、视频大小/修改时间或标注内容
  变化时都会失效并自动重算，避免误用旧缓存。
- **Subject/session** 分别表示用户和一次采集批次。真实测试目标是未见用户，因此
  `split.strategy` 默认是 `subject`，同一用户绝不会跨 train/validation/test。

当前目录名不能可靠提供 subject。复制
[`data/subject_mapping.example.csv`](data/subject_mapping.example.csv) 为
`data/subject_mapping.csv` 并填写真实映射。若缺失，prepare 会生成
`data/processed/missing_subject_mapping.csv` 后停止，不会把视频级结果误称为新用户
泛化。确实只能做视频级实验时，必须显式设置：

```yaml
split:
  strategy: video
```

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
