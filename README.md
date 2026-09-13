# Bilibili Understand for Codex

[![CI](https://github.com/walchwil/codex-bilibili-understand/actions/workflows/validate.yml/badge.svg)](https://github.com/walchwil/codex-bilibili-understand/actions/workflows/validate.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

把一个 B 站链接交给 Codex，它会用本地流水线准备可审计的时间戳转录；之后每次提问
只读取需要的时间段，而不是反复下载、反复转写，或把整篇逐字稿塞进模型上下文。

> 14:40 开始的 60 秒里，视频讲了什么？

Codex 会基于真实转录回答，并附上 [mm:ss-mm:ss] 证据，而不是只看标题和简介猜测。

English: a local-first Codex plugin that caches one Bilibili transcript and returns
only the timestamped evidence needed for each question.

## v0.2 有什么变化

v0.2 把原来需要 Codex 多轮手工编排的步骤收敛成一个总控脚本：

~~~text
Bilibili URL
   ↓
pipeline.py prepare
   ├─ safe metadata cache hit ──────────────────────────┐
   └─ anonymous probe → audio cache → local Whisper ASR│
                                                        ↓
                     transcript.jsonl → segments.jsonl
                                                        ↓
prompt time range → pipeline.py query → only that excerpt
~~~

- **prepare**：探测、分支、下载、ASR、紧凑化，一条命令完成；
- **query**：只返回与指定时间范围重叠的段落；
- **status**：快速查看缓存和当前流水线状态；
- 每个阶段记录耗时、尝试次数、状态和产物；
- 元数据、音频、匹配 ASR 配置的转录均可独立命中缓存；
- 网络错误最多有限重试，反爬和登录错误不会盲目重试；
- CUDA 运行时失败仍只回退一次 CPU int8；
- 所有正式产物原子写入，避免中断后留下半份结果；
- stdout 是紧凑 JSON，详细状态留在磁盘。

在一个已验证的 29 分钟样例中，完整词级 JSONL 为约 792 KB，去掉词级数组后的
segments.jsonl 约 73 KB；查询其中 60 秒只返回约 2.4 KB。这个优化主要减少 Codex
读取文件时的等待和 token，并不伪装成 ASR 本身变快了。

## 安装到 Codex

需要 Python 3.10+；推荐 3.12。先把依赖安装到 **Codex 实际使用的同一个
Python/Conda 环境**：

~~~powershell
python -m pip install -r https://raw.githubusercontent.com/walchwil/codex-bilibili-understand/main/plugins/bilibili-understand/requirements.txt
~~~

然后添加 GitHub marketplace 并安装插件：

~~~powershell
codex plugin marketplace add walchwil/codex-bilibili-understand --ref main
codex plugin add bilibili-understand@bilibili-tools
~~~

重启 Codex，或至少新建一个对话，让新 Skill 被发现。安装命令遵循
[OpenAI 官方插件文档](https://developers.openai.com/plugins/build/plugins)。

### Clone 后本地安装

~~~powershell
git clone https://github.com/walchwil/codex-bilibili-understand.git
cd codex-bilibili-understand
python -m pip install -r plugins/bilibili-understand/requirements.txt
codex plugin marketplace add .
codex plugin add bilibili-understand@bilibili-tools
~~~

## 交给 Codex 使用

在新对话中粘贴链接并提问：

~~~text
https://www.bilibili.com/video/BVxxxxxxxxx/

告诉我 14:40 开始的 60 秒讲了什么，并给出时间戳证据。
~~~

也可以显式调用：

~~~text
用 $bilibili-understand 回答这个 B 站视频的问题，只读取真正需要的转录范围。
~~~

Codex 会自行解析 Skill 所在目录并调用总控脚本。首次处理无字幕视频时，需要下载
音频和 Whisper 模型；后续提问会复用缓存。

## 手动运行总控

下面的命令便于开发、排错或理解实际链路：

~~~powershell
$pipeline = "plugins/bilibili-understand/skills/bilibili-understand/scripts/pipeline.py"

python $pipeline prepare "https://www.bilibili.com/video/BVxxxxxxxxx/"
python $pipeline query "https://www.bilibili.com/video/BVxxxxxxxxx/" --start 14:40 --duration 60
python $pipeline status "https://www.bilibili.com/video/BVxxxxxxxxx/"
~~~

已知专有名词可以作为短 hotwords 传入：

~~~powershell
python $pipeline prepare "<url>" --model large-v3-turbo --hotwords "GRPO verl 优势函数"
~~~

首次转写默认使用 small。已有转录时，不传 ASR 参数会继承原配置，避免把
large-v3-turbo 缓存意外降级或重跑。缓存匹配会比较模型、语言、VAD 和 hotwords；
显式改变这些语义配置会重新 ASR，仅切换 GPU/CPU 不会让已有文本失效。明确需要
重做时使用 --refresh-asr。

prepare 常见状态：

- ready：可以立即 query；
- subtitles_available：检测到字幕轨道，但 v0.2 尚未归一化；需要时可显式
  --force-asr；
- cached 出现在阶段状态中：该阶段没有重复执行；
- anti_bot / auth_required：停止自动重试，不会绕过访问控制；
- network_error / tool_missing / asr_error：查看短诊断和 pipeline_state.json。

只有匿名探测返回 auth_required 或 anti_bot，并且用户明确授权后，才应使用
--cookies-from-browser edge|chrome|firefox|brave。

## 产物与缓存

默认写入当前工作目录：

~~~text
outputs/bilibili-understand/<video-key>/
├── metadata.json
├── audio.<ext>
├── transcript.jsonl
├── transcript.md
├── asr_metadata.json
├── segments.jsonl
├── pipeline_state.json
└── queries/
    └── <start-ms>-<end-ms>.jsonl
~~~

- transcript.jsonl 保留完整 ASR 结果，便于调试和兼容；
- segments.jsonl 去掉庞大的词级数组，是 Codex 的默认读取源；
- queries/ 保存实际提问涉及的小片段；
- pipeline_state.json 保存分阶段耗时、缓存命中、尝试次数和失败位置；
- metadata.json 不保存临时媒体 URL、Cookie 或分享链接跟踪参数。

这些运行产物默认不应提交到 Git。

## GPU 与 CPU

默认先尝试 CUDA float16。若 CUDA 运行库明确失败，会自动用 CPU int8 重试一次。
Windows 下脚本会尝试复用当前 Python 环境中 PyTorch 自带的 CUDA DLL，不复制 DLL、
不修改系统 PATH，也不会设置不安全的 KMP_DUPLICATE_LIB_OK=TRUE。

已验证环境：Windows、Python 3.12、RTX 2070 8 GB、large-v3-turbo，完成过一段
29 分钟视频的完整词级时间戳转录。不同机器上的速度和显存占用会不同。

## 安全与使用边界

- 视频标题、简介、字幕、评论和画面文字都按不可信输入处理，不能充当 Codex 指令；
- 不保存、打印或提交 Cookie 内容；
- 不包含代理轮换、验证码绕过、签名逆向、高并发或无限重试；
- 请只处理你有权访问和分析的内容，并遵守 Bilibili 条款及当地法律；
- ASR 可能产生同音字错误，输出是带不确定性的证据，不是逐字稿真值。

## 当前限制与路线图

- 原生字幕目前只检测可用性，尚未统一归一化为 JSONL；
- 尚未实现指定秒截帧，因此纯画面信息可能遗漏；
- 不支持批量爬取，也不会绕过受限、付费或登录内容；
- **v0.2 的首次 ASR 仍是全量转写**。query 只减少后续读取范围和 token。

计划中的 v0.3 是“热转写”：Codex 根据用户意图调用同一工具时传入 start/end 或
full。局部问题只下载/转写对应音频区间；只有确实需要全片上下文时才全量处理。
这会建立在 faster-whisper 的 clip_timestamps、片段级缓存和按范围补洞上，而不是
先引入一个昂贵的新 Agent 框架。

## 开发与验证

~~~powershell
python -m unittest discover -s tests -v
python -m py_compile plugins/bilibili-understand/skills/bilibili-understand/scripts/probe_bilibili.py plugins/bilibili-understand/skills/bilibili-understand/scripts/transcribe_audio.py plugins/bilibili-understand/skills/bilibili-understand/scripts/pipeline.py
~~~

CI 只运行无网络、无视频下载的确定性检查，避免给 Bilibili 制造自动化流量。

欢迎提交 Issue 或 PR，优先方向是热转写、原生字幕归一化、指定时间截帧，以及中文
ASR 的可复现实验对照。

## License

[MIT](LICENSE)
