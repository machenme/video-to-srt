# ASR Pipeline — 视频转日语文本并行处理管线

基于 `faster-whisper`（CTranslate2 后端）的离线批量视频语音转文字工具。利用本地 GPU 多路并行推理，支持长视频自动切割、并行转写、分段合并，一键输出 SRT 字幕、纯文本和 Markdown 文件。

## 特性

- **GPU 多路并行** — spawn 独立进程，每进程常驻一个 WhisperModel，信号量控制显存
- **长视频自动切割** — 超过 N 分钟的音频自动切成片段，多 Worker 并行处理，最后合并时间戳
- **断点续跑** — 中断后重跑自动跳过已完成的视频，进度持久化到 `.progress.json`
- **字幕级切分** — SRT 输出按标点 + 时长切分为可读短句（2-7 秒 / 条，≤40 字）
- **三种输出** — SRT 字幕 + TXT 纯文本 + MD 带时间轴，各司其职
- **全离线** — 模型本地加载，无网络依赖，零数据外泄

## 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA 8 GB VRAM | RTX 5070 Ti 16 GB |
| NVIDIA 驱动 | ≥535 | ≥545 |
| 内存 | 16 GB | 32 GB |
| 存储 | SSD | NVMe SSD（临时音频 I/O） |
| OS | Windows 11 / Ubuntu 22.04+ | |

## 环境准备

### 1. 安装 ffmpeg

```bash
# Windows (scoop)
scoop install ffmpeg

# 验证
ffmpeg -version
```

### 2. Python 环境

使用 `mise` → `uv` → Python 3.11 工具链：

```bash
cd video-to-text
uv python pin 3.11          # 固定 Python 3.11
uv venv                      # 创建虚拟环境
uv pip install -r requirements.txt
```

### 3. 下载模型

从 HuggingFace 克隆 `faster-whisper-large-v3-turbo-ct2`（CT2 格式，开箱即用）：

```bash
mkdir models
cd models
git lfs install
git clone https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2
```

> 单实例 FP16 约 2.5 GB 显存。RTX 5070 Ti 16 GB 可跑 **4 路并行**，余量充足。

## 快速开始

```bash
# 处理单个视频
uv run python -m src.main --input . --output ./output

# 输出结构：
#   output/
#   └── demo1/                ← 每个视频一个子文件夹
#       ├── demo1.srt         ← 标准 SRT 字幕
#       ├── demo1.txt         ← 纯文本（无时间戳）
#       └── demo1.md          ← Markdown 带 [HH:MM:SS] 时间轴
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | PATH | **必填** | 输入视频目录 |
| `--output` | PATH | **必填** | 输出根目录 |
| `--config` | PATH | `./config.yaml` | 配置文件路径 |
| `--model` | str | `large-v3-turbo` | 模型：large-v3-turbo / large-v3 / medium |
| `--workers` | int | 4 | 最大并行 Worker 数 |
| `--chunk-duration` | int | 900 | 长视频切割阈值（秒），0 = 禁用 |
| `--temp-dir` | PATH | 系统临时目录 | 临时音频存放路径 |
| `--language` | str | `ja` | 目标语言（ISO 639-1） |
| `--beam-size` | int | 5 | Beam Search 宽度 (1-10) |
| `--compute-type` | str | `float16` | 推理精度：float16 / int8_float16 |
| `--no-vad` | flag | false | 禁用 VAD 语音检测 |
| `--no-cleanup` | flag | false | 保留临时音频文件 |
| `--verbose` | flag | false | 输出 DEBUG 级日志 |
| `--force` | flag | false | 忽略断点续跑，强制全部重跑 |

CLI 参数优先级高于配置文件。

## 配置文件

项目根目录的 `config.yaml`，可通过 `--config` 指定自定义路径：

```yaml
input_dir: "."                     # 输入视频目录
output_dir: "./output"             # 输出根目录
temp_dir: null                     # 临时音频（null = 系统临时目录）

model_path: "./models/faster-whisper-large-v3-turbo-ct2"
model_size: "large-v3-turbo"
language: "ja"
beam_size: 5
vad_filter: true
compute_type: "float16"

max_workers: 4                     # GPU 并行数
chunk_duration: 900                # 长视频切割阈值（秒），0 = 禁用

video_extensions:                  # 扫描的扩展名
  - mp4
  - mkv
  - mov
  - avi
  - flv
  - wmv

output_formats:                    # 输出格式
  - srt
  - txt
  - md

cleanup_temp: true                 # 完成后清理临时文件
```

## 使用示例

### 批量处理

```bash
uv run python -m src.main --input ./videos --output ./subtitles
```

### 调整并发和切割策略

```bash
# 6 路并行 + 10 分钟切割
uv run python -m src.main --input ./videos --output ./out --workers 6 --chunk-duration 600

# 禁用切割（短视频场景）
uv run python -m src.main --input ./clips --output ./out --chunk-duration 0
```

### 断点续跑

```bash
# 中断后重新执行相同命令，自动跳过已完成视频
uv run python -m src.main --input ./videos --output ./subtitles

# 强制重跑全部
uv run python -m src.main --input ./videos --output ./subtitles --force
```

### 切换语言

```bash
uv run python -m src.main --input ./videos --output ./out --language en
```

## 管线架构

```
[视频扫描] → [ffmpeg 提取 16kHz Mono WAV] → [音频切割 (可选)]
                                                    │
                                          ┌─────────┼─────────┐
                                          ▼         ▼         ▼
                                     [Worker 0] [Worker 1] [Worker 2] ...
                                     (WhisperModel 常驻显存，GPU 并行推理)
                                          │         │         │
                                          └─────────┼─────────┘
                                                    ▼
                                          [分段合并 + 时间偏移]
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                output/           output/         output/
                              video_a/          video_b/        video_c/
                            ├── .srt           ├── .srt        ├── .srt
                            ├── .txt           ├── .txt        ├── .txt
                            └── .md            └── .md         └── .md
```

### 长视频切割机制

对于超过 `chunk_duration` 秒的视频：用 ffmpeg segment muxer 将 WAV 切成固定时长片段 → 所有片段作为独立任务送入 GPU 调度器并行处理 → 每个片段返回带相对时间戳的 segments → `combine_chunk_segments()` 根据片段偏移量还原绝对时间 → 跨 chunk 边界的重叠段自动去重合并。

不切割时（视频短或 `chunk_duration: 0`），整个 WAV 作为一个任务直接转写。

## 输出格式

### SRT（标准字幕）

```
1
00:00:18,000 --> 00:00:20,000
時間ないです。今日はありがとうございます。

2
00:00:20,000 --> 00:00:24,000
フラエティーに参加していただいて、料金ゲットっていうのをやってるんですけど、
```

每条 2-7 秒、≤40 字符，按句尾标点切分，适合直接嵌入视频。

### TXT（纯文本）

日文无空格连续拼接，适合全文搜索和 NLP 下游处理。

### MD（Markdown 带时间轴）

```markdown
[00:00:18] 時間ないです。今日はありがとうございます。
[00:00:20] フラエティーに参加していただいて、料金ゲットっていうのをやってるんですけど、
```

时间戳精确到秒，方便人工校对定位。

## 实测性能（RTX 5070 Ti 16 GB）

| 视频时长 | 切割 | Chunk 数 | Worker | 处理耗时 | 实时倍数 |
|----------|------|----------|--------|----------|----------|
| 29 分钟 | 15 min | 2 | 4 | ~27 秒 | ~63x |
| 131 分钟 | 15 min | 9 | 4 | ~95 秒 | ~**82x** |
| 270 分钟（估） | 15 min | 18 | 4 | ~3 分钟 | ~90x |

> 实时倍数 = 视频时长 / 处理耗时。倍数随视频增长而上升，因为 GPU 并行度被长视频切割填满。

## 退出码

| Code | 含义 |
|------|------|
| 0 | 全部任务成功 |
| 1 | 参数错误或配置无效 |
| 2 | 部分任务失败 |
| 3 | 致命错误（GPU 不可用 / OOM） |

## 项目结构

```
video-to-text/
├── src/
│   ├── main.py              # CLI 入口，流程编排，信号处理
│   ├── config.py            # YAML 配置加载 / CLI 覆盖 / 校验
│   ├── audio_extractor.py   # Stage 1: ffmpeg 提取 + 切割 + 容错 fallback
│   ├── gpu_scheduler.py     # Stage 2: spawn 多进程 + 信号量 GPU 调度
│   ├── transcribe_worker.py # Stage 2: faster-whisper 推理 + 模型缓存
│   ├── text_formatter.py    # Stage 3: 分段合并 + 字幕切分 + 格式化
│   ├── task_manager.py      # 任务状态跟踪 + 断点续跑
│   ├── monitor.py           # GPU 显存 / 利用率监控
│   └── utils.py             # 文件扫描、SRT 校验、时间戳格式化
├── models/                  # 模型文件（需自行下载）
│   └── faster-whisper-large-v3-turbo-ct2/
├── output/                  # 输出目录
│   └── {video_name}/
│       ├── {video_name}.srt
│       ├── {video_name}.txt
│       └── {video_name}.md
├── config.yaml              # 默认配置
├── requirements.txt         # Python 依赖
├── pyproject.toml           # uv 项目配置
├── .gitignore
└── README.md
```

## 常见问题

**Q: 报错 `cublas64_12.dll is not found`？**

```bash
uv pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

已包含在 `requirements.txt` 中。若仍报错，确认 CUDA 驱动版本 ≥535。

**Q: 显存不足 OOM？**

降低 `--workers`。turbo 单实例约 2.5 GB，按 `可用显存 / 2.5` 计算安全值。

**Q: 音视频不同步？**

ffmpeg 提取时若遇损坏 AAC 流会自动触发 raw-AAC fallback（两步法：先裸流拷贝，再独立解码）。若仍失败，源文件音频轨道可能严重损坏。

**Q: SRT 字幕太长 / 太短？**

调整 `config.yaml` 的 `chunk_duration`（影响切割粒度）和编辑 `text_formatter.py` 中的 `_MAX_CHARS_PER_SUB`（默认 40）和 `_MAX_SUB_DURATION`（默认 7.0 秒）。

**Q: 字幕时间轴对不上视频？**

子进程 CUDA context 隔离正常时不应出现。若发生，检查 `--chunk-duration` 是否过小导致 chunk 边界过多，或源视频帧率异常。
