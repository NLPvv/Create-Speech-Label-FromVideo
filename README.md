# VideoSubtitleExtractor

从视频固定区域的硬字幕生成 SRT：抽帧/裁剪 → PP-OCRv5 单行文字识别 → 连续字幕合并。适合字幕位置固定的视频；ROI 应尽量只包含一行字幕，当前不包含文字检测或多行版面分析。

## 安装（Conda）

先安装 Miniconda 或 Anaconda，以下命令在项目根目录执行。使用 Conda 创建独立的 Python 3.10 环境，并安装 FFmpeg（包含 `ffprobe`）；Python 依赖通过该环境内的 pip 安装。PaddlePaddle 的 CPU/GPU 运行时单独选择安装。

```bash
# 创建并激活环境
conda create -n video-subtitle -c conda-forge python=3.10 pip ffmpeg -y
conda activate video-subtitle

# 安装项目 Python 依赖
python -m pip install -r requirements.txt

# 以下 CPU / GPU 运行时二选一
# CPU 示例
python -m pip install paddlepaddle==3.2.2

# 或：Linux NVIDIA GPU / CUDA 11.8 示例（与 CPU 版二选一）
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 验证安装
ffmpeg -version
ffprobe -version
python -c 'import paddle; paddle.utils.run_check()'
```

后续运行项目前，先执行 `conda activate video-subtitle`。环境管理说明见 [Conda 官方文档](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。

具体平台和 CUDA 组合请参考 [PaddlePaddle 官方安装说明](https://www.paddlepaddle.org.cn/install/quick)；文字识别接口参考 [PaddleOCR 官方文档](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html)。首次识别可能需要下载模型。`--hpi` 需要额外的推理环境，默认关闭。

`PyTurboJPEG` 是可选 JPEG 解码加速依赖，仅 OpenCV 裁剪流程使用；未安装 Python 包或原生 `libturbojpeg` 时会自动回退 OpenCV。可按需安装 `PyTurboJPEG==1.7.7`。

## 快速开始

所有命令均在项目根目录运行。ROI 顺序统一为 `XMIN YMIN XMAX YMAX`，右/下边界不包含在区域内，坐标使用视频显示画面的像素。

```bash
python main.py /path/to/video.mp4 \
  --roi 320 590 1280 711 \
  --output output/video.srt \
  --device gpu:0 --batch-size 5 --fps 25

# 查看参数；无需先加载 OCR 模型
python main.py --help
```

默认直接在 FFmpeg 抽帧时裁剪，只写字幕区域图片，减少磁盘写入、JPEG 再解码和二次压缩。需要完整帧时可加 `--keep-full-frames --workers 4`，改用独立的 OpenCV 裁剪步骤。默认设备为 `cpu`，不指定 `--output` 时 SRT 写在视频旁。

| 参数                 | 默认值                 | 作用                                               |
| ------------------ | ------------------- | ------------------------------------------------ |
| `--fps`            | 25                  | 抽样帧率，同时用于 SRT 时间换算；降低可减少 OCR 工作量，但降低时间精度且可能漏过短字幕 |
| `--score`          | 0.9                 | OCR 最低置信度                                        |
| `--sim-threshold`  | 80                  | 合并文本所需的相似度，范围 0–100                              |
| `--max-gap-frames` | 1                   | 同一段字幕中相邻 OCR 记录的最大帧号差；2 可容忍缺失一帧                  |
| `--model-name`     | PP-OCRv5_server_rec | 文字识别模型名                                          |
| `--work-dir`       | work                | 中间图片及可选 JSON 的父目录                                |
| `--save-json`      | 关闭                  | 按视频/运行目录保存 OCR JSON                              |
| `--verbose`        | 关闭                  | 打印逐帧识别结果及详细错误                                    |

抽帧目录带 `.frames-complete.json` 完成记录，只有源视频签名、参数及连续非空帧检查一致才会复用。旧目录、未完成结果及不同参数的结果保留，新运行使用独立目录。抽帧或 OCR 失败会返回非零退出码；全部文本未达到置信度时会提示并生成空 SRT。

字幕默认按 FFmpeg 的一基帧号计时，第 1 帧从 `00:00:00,000` 开始。相同文字隔着缺帧默认分开，避免把无字幕片段合并进去。已有外部图片若从 0 编号，可在 Python API 中使用 `SubtitleGenerator(..., frame_index_base=0)`。

## 批量处理

`videos.txt` 是 UTF-8 文本，每行一个视频路径。批量抽帧复用同一实现，可直接裁剪：

```bash
python -m utils.batch_ffmpeg videos.txt \
  --output-root frames --fps 25 --nj 4 \
  --roi 320 590 1280 711

# 直接识别上一步的所有视频帧目录，多 GPU 各复用一个模型
python -m utils.batch_main_ocr_sub frames \
  --suffix '' --output-dir output --gpus 0 1 \
  --batch-size 10 --fps 25
```

如果已有完整帧目录，可单独裁剪后识别：

```bash
python -m utils.batch_main_cut frames \
  --roi 320 590 1280 711 --output-root cropped --num-workers 4

python -m utils.batch_main_ocr_sub cropped \
  --output-dir output --gpus 0 1 --fps 25 --save-json
```

批裁剪跳过已有 `*_crop` 目录，并保留全部源文件。目标目录若含当前输入之外的图片，会拒绝继续写入；请选择新的输出目录。批 OCR 默认只处理 `*_crop` 目录，移除末尾 `_crop` 生成 SRT 文件名。不传 `--gpus` 时使用一个进程，可通过 `--device cpu` 或 `--device gpu:0` 指定设备。识别阶段的 `--fps` 必须与图片抽样帧率一致；请为一次批任务使用统一帧率，或按帧率分组。

## 代码结构与验证

```text
core/frame_skip.py          FFmpeg 抽帧、直接裁剪和完成记录
core/cropper.py             图片裁剪、可选 TurboJPEG 加速
core/ocr_module.py          OCR 批识别、置信度过滤、模型复用
core/subtitle_generator.py  字幕合并和 SRT 时间计算
main.py                    单视频命令行入口
utils/                     批量处理入口
tests/                     无需 GPU 的 unittest 回归测试
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

测试覆盖目录衔接、FFmpeg 失败和缓存校验、裁剪失败和旧帧混入、OCR 模型复用与过滤、字幕计时和缺帧边界。推理与图像库接口使用替身，不验证模型识别精度；真实推理仍需在目标运行环境用代表性视频验证。测试临时文件会保留，不自动清理。

## Pipeline

![pipeline.png](images/pipeline.png)

### 音视频抽帧（Frame Extraction）

视频会根据设定的 FPS 进行抽帧，将连续的视频流拆分成等间隔的图片帧，便于后续进行 OCR 与字幕生成。

![frame.png](images/frame.png)

### 区域裁剪（Crop Video）

由于视频中通常只有部分区域包含字幕，本步骤会自动裁剪出你定义的文本区域（ROI），减少 OCR 输入规模，提高识别精度和速度。

![crop.png](images/crop.png)

### OCR 文本识别（PaddleOCR v5）

通过 PaddleOCR 的 OCR 模型对裁剪后的字幕区域进行逐帧识别，输出按时间顺序排列的识别结果。

```md
1011/25 观众朋友们大家好 100
1012/25 观众朋友们大家好 100
1013/25 观众朋友们大家好 100
1014/25 观众朋友们大家好 100
1015/25 观众朋友们大家好 100
1016/25 观众朋友们大家好 100
1017/25 观众朋友们大家好 100
1018/25 观众朋友们大家好 100
1019/25 观众朋友们大家好 100
1020/25 欢迎您收看《壮语新闻》我是黄利莉 0
1021/25 欢迎您收看《壮语新闻》我是黄利莉 100
1022/25 欢迎您收看《壮语新闻》我是黄利莉 100
1023/25 欢迎您收看《壮语新闻》我是黄利莉 100
1024/25 欢迎您收看《壮语新闻》我是黄利莉 100
1025/25 欢迎您收看《壮语新闻》我是黄利莉 100
1026/25 欢迎您收看《壮语新闻》我是黄利莉 100
1027/25 欢迎您收看《壮语新闻》我是黄利莉 100
1028/25 欢迎您收看《壮语新闻》我是黄利`梨` 94
......
```

### 字幕文件合并（Subtitle Merge）

OCR 输出的逐帧文本会经过去噪、基于文本相似度算法将重复内容合并，并基于帧率生成精确时间戳，最终自动整合同一内容的连续帧并导出为 `.srt` 字幕文件用于后续 ASR 训练。

#### 示例输出（srt）：

```lua
1
00:00:39,000 --> 00:00:40,920
观众朋友们大家好

2
00:00:40,920 --> 00:00:43,719
欢迎您收看《壮语新闻》我是黄利莉

3
00:00:43,759 --> 00:00:46,079
面对开放发展带来的

4
00:00:46,079 --> 00:00:48,439
出入境人数持续增长新形势

......
```
