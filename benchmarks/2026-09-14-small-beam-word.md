# ASR 配置对照：small / beam size / 词级时间戳

## 实验目标

固定同一音频、同一语言、同一模型和同一时间窗口，只改变 `beam_size` 与
`word_timestamps`，观察耗时、输出体积和文本差异。

## 固定条件

- 音频：`BV1rrVd6fEsm/audio.m4a`
- 目标问题：`14:40-15:40`
- 实际 ASR 窗口：`877-943s`（前后各 3 秒）
- 模型：`small`
- 设备：RTX 2070 Max-Q / CUDA / float16
- 语言：`zh`
- VAD：关闭
- hotwords：无
- 每组独立 Python 进程，模型文件复用本地缓存

复现实验命令（把 `<variant>` 替换为输出目录）：

```powershell
python transcribe_audio.py `
  "<audio.m4a>" `
  --output-dir "<variant>" `
  --model small `
  --model-cache "<model-cache>" `
  --language zh `
  --device cuda `
  --compute-type float16 `
  --beam-size 5 `
  --clip-start 877 `
  --clip-end 943
```

词级组额外增加 `--word-timestamps`，beam3 组把 `--beam-size` 改为 `3`。

## 结果

| 配置 | 第 1 次 model/decode (s) | 第 2 次 model/decode (s) | 段数 | 词数 | transcript 大小 |
|---|---:|---:|---:|---:|---:|
| beam5 + segment | 4.70 / 11.24 | 3.69 / 10.32 | 35 | 0 | 4.1 KB |
| beam3 + segment | 4.09 / 9.86 | 3.26 / 18.42 | 35 | 0 | 4.1 KB |
| beam5 + word | 3.37 / 11.24 | 4.80 / 19.88 | 35 | 319 | 28.2 KB |
| beam3 + word | 3.94 / 10.21 | 4.00 / 19.06 | 35 | 319 | 28.2 KB |

## 结论

1. 词级时间戳把 transcript 从约 4.1 KB 增加到 28.2 KB，约 6.9 倍；它是存储和
   Codex 读取成本上的确定性开销。当前没有证据表明这段音频的解码时间必然增加，
   所以继续保持“段级默认、词级按需”。
2. beam3 第一轮较快，但第二轮明显变慢；两轮结果不足以证明 beam3 优于 beam5。
   当前不把 beam3 设成全局默认。
3. 两个 beam 的段数都为 35，但文本存在少量差异：segment 组约 2/35 段不同，
   word 组约 1/35 段不同。示例包括“匹配/部匹配”“简历/减力”等中文识别差异。
4. `model_load_s` 约 3.3–4.8 秒，`decode_s` 约 9.9–19.9 秒。模型冷启动不是全部
   成本，但值得在更多请求下测量；真正的常驻 worker 需要先做随机化顺序、GPU 预热、
   多轮重复和显存/温度记录。

## 当前工程决策

- 默认：`beam_size=5`、不保留词级时间戳；这是保守、可解释的默认值。
- 需要逐词引用时：显式使用 `--word-timestamps`。
- 想验证 beam3：只在固定数据集上做对照，不凭一次运行改默认值。
- 下一轮性能实验应加入预热、随机化执行顺序、至少 5 次重复，并记录 GPU 温度、功耗、
  利用率和显存；否则容易把 GPU 时钟波动误判成算法收益。
