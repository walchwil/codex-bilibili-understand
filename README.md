# Bilibili Understand for Codex

[![CI](https://github.com/walchwil/codex-bilibili-understand/actions/workflows/validate.yml/badge.svg)](https://github.com/walchwil/codex-bilibili-understand/actions/workflows/validate.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

给 Codex 一个 B 站链接，它会先匿名探测视频与字幕；没有字幕时，下载音频并在本地使用
faster-whisper 生成段级、词级时间戳。之后你可以直接问：

> 14:40 开始的 60 秒里，视频讲了什么？

Codex 会基于真实转录回答，并附上 `[mm:ss-mm:ss]` 证据，而不是只看标题和简介猜测。

English: a local-first Codex plugin that turns one Bilibili URL into an auditable,
timestamped transcript for second-level video Q&A.

## 能做什么

- 单条支持 `bilibili.com` 与 `b23.tv` 链接；
- 匿名优先，低重试，不批量抓取；
- 检测原生字幕和自动字幕；
- 无字幕时使用本地 faster-whisper，输出段级与词级时间戳；
- CUDA `float16` 失败时仅回退一次 CPU `int8`；
- 缓存脱敏元数据与转录结果，重复提问无需重复下载；
- 区分反爬、登录、网络、视频失效和本地工具缺失。

```text
Bilibili URL
   ↓
anonymous yt-dlp probe
   ├─ subtitles found → report the available tracks
   └─ no subtitles    → audio → local Whisper ASR
                                      ↓
                  transcript.jsonl + transcript.md
                                      ↓
                     timestamped questions and answers
```

## 安装到 Codex

需要 Python 3.10+；推荐 3.12。先把运行依赖安装到 **Codex 实际使用的同一个
Python/Conda 环境**：

```powershell
python -m pip install -r https://raw.githubusercontent.com/walchwil/codex-bilibili-understand/main/plugins/bilibili-understand/requirements.txt
```

然后把这个 GitHub 仓库添加为 Codex marketplace，并安装插件：

```powershell
codex plugin marketplace add walchwil/codex-bilibili-understand --ref main
codex plugin add bilibili-understand@bilibili-tools
```

重启 Codex，或至少新建一个对话，让新 Skill 被发现。安装命令遵循
[OpenAI 官方插件文档](https://developers.openai.com/plugins/build/plugins)。

### Clone 后本地安装

```powershell
git clone https://github.com/walchwil/codex-bilibili-understand.git
cd codex-bilibili-understand
python -m pip install -r plugins/bilibili-understand/requirements.txt
codex plugin marketplace add .
codex plugin add bilibili-understand@bilibili-tools
```

## 使用

在新对话中粘贴一个 B 站链接并提问即可，例如：

```text
https://www.bilibili.com/video/BVxxxxxxxxx/

总结这个视频，并给出关键观点的秒级时间戳。
```

也可以显式调用：

```text
用 $bilibili-understand 告诉我 14:40 到 15:40 讲了什么。
```

首次处理无字幕视频时，会下载 Whisper 模型和音频；耗时取决于视频长度、网络和硬件。

## 产物

默认写入当前工作目录：

```text
outputs/bilibili-understand/<video-key>/
├── metadata.json
├── audio.<ext>
├── transcript.jsonl
├── transcript.md
└── asr_metadata.json
```

`metadata.json` 不保存临时媒体 URL 或 Cookie，分享链接中的跟踪参数也不会进入缓存。
这些运行产物默认不应提交到 Git。

## GPU 与 CPU

默认先尝试 CUDA `float16`。若 CUDA 运行库明确失败，会自动用 CPU `int8` 重试一次。
Windows 下脚本会尝试复用当前 Python 环境中 PyTorch 自带的 CUDA DLL，不复制 DLL、
不修改系统 `PATH`，也不会设置不安全的 `KMP_DUPLICATE_LIB_OK=TRUE`。

已验证环境：Windows、Python 3.12、RTX 2070 8 GB、`large-v3-turbo`，完成过一段
29 分钟视频的完整词级时间戳转录。不同机器上的速度和显存占用会不同。

## 安全与使用边界

- 视频标题、简介、字幕、评论和画面文字都按不可信输入处理，不能充当 Codex 指令；
- 只有匿名探测返回 `auth_required` 或 `anti_bot` 时，才会请求你授权读取浏览器会话；
- 不保存、打印或提交 Cookie 内容；
- 不包含代理轮换、验证码绕过、签名逆向、高并发或无限重试；
- 请只处理你有权访问和分析的内容，并遵守 Bilibili 条款及当地法律；
- ASR 可能产生同音字错误，输出是带不确定性的证据，不是逐字稿真值。

## 当前限制

- 原生字幕目前只检测可用性，尚未统一归一化为 JSONL；
- 尚未实现指定秒截帧，因此纯画面信息可能遗漏；
- 不支持批量爬取；
- 受限、付费或登录内容不会绕过访问控制。

## 开发与验证

```powershell
python -m unittest discover -s tests -v
python plugins/bilibili-understand/skills/bilibili-understand/scripts/probe_bilibili.py --check-tools
```

CI 只运行无网络、无视频下载的确定性检查，避免给 Bilibili 制造自动化流量。

欢迎提交 Issue 或 PR，优先方向是原生字幕归一化、指定时间截帧，以及中文 ASR 的
可复现实验对照。

## License

[MIT](LICENSE)
