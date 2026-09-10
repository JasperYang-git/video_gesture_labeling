# Video Gesture Labeling

离线摄像头手势时序分割 Pipeline。输入一段手势视频，模型输出逐帧动作类别（14 种手势 + background）。

当前仓库按配置驱动的方式重建了训练和推理流程。真实训练视频在另一台服务器上且无法访问，因此仓库内置确定性模拟数据，用来打通 `prepare → train → infer → evaluate`。MediaPipe 真实视频入口已经保留，拿到数据后只需关闭 mock 并指向原始目录。

## 任务与特征

- 任务：Temporal Action Segmentation，整段视频离线预测
- 模型：MS-TCN2（Prediction Generation + Refinement）
- 特征：128 维 `score + v_mask + 63 坐标 + 63 速度`
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

训练、验证、测试按原始视频划分，避免重叠窗口泄漏。

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

真实路径依赖 `opencv-python` 和 `mediapipe`。预处理缓存带配置指纹，改 FPS、平滑参数或左右手后不会静默复用旧缓存。

左右手按文件夹名第 3 段和 MediaPipe handedness 同时过滤，不再只取置信度最高的手。

## 输出

- 训练：`outputs/training_results/MMDD/MMDD_HHMMSS/`，最优权重另存为 `outputs/best_model.pth`
- 推理：每个视频的 `frames.csv`、`prediction.npy`、`segments.json`
- 评估：`outputs/evaluation_results/.../metrics.json`

Checkpoint 自带模型名、参数、类别映射和特征 schema，推理不需要再声明模型结构。

## 测试

```bash
python -m unittest discover -s tests -v
```

单元测试覆盖标注解析、视频级划分、窗口掩码、模型形状、损失、重叠拼接和指标。MediaPipe 检出质量与真实时间对齐只有在拿到可运行的加密数据后才能验证。
