# AICARE - 自动语音识别（ASR）系统

## 项目简介

AICARE 是一个基于 **OpenAI Whisper** 模型构建的实时和批量语音识别系统，用于语音转文字转录。系统提供：

- **实时音频测试**：捕获实时麦克风输入并实时转录语音
- **批量音频测试**：从 HuggingFace 数据集并发处理多个音频样本
- **网页前端界面**：交互式界面控制和监控 ASR 任务
- **可扩展架构**：WebSocket + RabbitMQ 消息队列实现分布式处理

**技术栈**：Python, FastAPI, WebSocket, RabbitMQ, PyTorch, Transformers (Whisper), PyAudio

---

## 快速开始指南

### 前置要求

**系统依赖：**
```bash
# 安装系统依赖（Ubuntu/Debian）
sudo apt update
sudo apt install -y portaudio19-dev erlang rabbitmq-server

# 启动 RabbitMQ 服务
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
```

### 安装步骤

**1. 创建虚拟环境：**
```bash
cd /path/to/useful-file
python3 -m venv asrvenv
source asrvenv/bin/activate  # Windows 系统：asrvenv\Scripts\activate
```

**2. 安装 Python 依赖：**
```bash
pip install -r requirements.txt
```

**3. 下载 ASR 模型（可选）：**
系统在首次运行时会**自动下载** Whisper-small 模型（如果不存在）。或手动下载：
```bash
# 模型将存储在：ASR_model/openai-whisper-small/
# 首次运行 ASR_server.py 会自动处理
```

---

## 运行系统

**按顺序在不同终端启动所有组件**：

### 终端 1：WebSocket 网关
```bash
source asrvenv/bin/activate
python ASR_websockets.py
```
输出：`WebSocket server started on ws://localhost:8765`

### 终端 2：ASR 模型服务器
```bash
source asrvenv/bin/activate
python ASR_server.py
```
输出：`Server side start listening...`

### 终端 3：前端 API 服务器
```bash
source asrvenv/bin/activate
python frontend_api.py
```
输出：`Uvicorn running on http://0.0.0.0:3006`

### 终端 4：访问 Web 界面
在浏览器中打开：
```
http://localhost:3006/frontend/index.html
```

**两个功能按钮：**
- **实时输入测试**：启动/停止实时麦克风转录
- **多输入测试**：运行 100 个并发音频样本的批量测试

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Browser)                           │
│                   http://localhost:3006                         │
│  ┌──────────────────┐          ┌──────────────────┐            │
│  │ Real Input Test  │          │ Multi-Input Test │            │
│  │    (Button)      │          │    (Button)      │            │
│  └────────┬─────────┘          └────────┬─────────┘            │
└───────────┼─────────────────────────────┼──────────────────────┘
            │                             │
            │ HTTP/REST                   │ HTTP/REST
            ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND API SERVER (FastAPI)                      │
│                    frontend_api.py                              │
│                      Port: 3006                                 │
│  • Subprocess management (real-time client)                     │
│  • Batch test orchestration                                     │
│  • Result polling endpoints                                     │
└───────────┬─────────────────────────────┬───────────────────────┘
            │                             │
            │ subprocess.Popen()          │ asyncio.create_task()
            ▼                             ▼
┌──────────────────────┐      ┌──────────────────────────────────┐
│  REAL-TIME CLIENT    │      │   BATCH TEST CLIENT              │
│ client_real_mimic    │      │ auto_dataset_client_mimic.py     │
│     _api.py          │      │ (100 concurrent requests)        │
└──────────┬───────────┘      └─────────────┬────────────────────┘
           │                                │
           │ PyAudio (16kHz, Float32)       │ HuggingFace Dataset
           │ 5s accumulation                │
           │ Base64 encode                  │
           │                                │
           └────────────┬───────────────────┘
                        │ WebSocket (async)
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│          WEBSOCKET GATEWAY (ASR Producer)                       │
│                 ASR_websockets.py                               │
│                   Port: 8765                                    │
│  • WebSocket server for audio streaming                         │
│  • RPC pattern with correlation_id                              │
│  • Publishes to RabbitMQ asr_queue                              │
│  • Receives responses via callback_queue                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ RabbitMQ (pika)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              MESSAGE QUEUE (RabbitMQ)                           │
│                  localhost:5672                                 │
│  Queue: asr_queue (durable, QoS prefetch=1)                     │
│  Pattern: RPC with correlation_id + reply_to                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Consume messages
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│             ASR SERVER (Model Consumer)                         │
│                   ASR_server.py                                 │
│  • Consumes from asr_queue                                      │
│  • Decodes Base64 audio → numpy array                           │
│  • Processes through Whisper model                              │
│  • Publishes results to reply_to queue                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Model inference
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              ASR MODEL (Whisper)                                │
│                   ASR_model.py                                  │
│  Model: openai/whisper-small                                    │
│  Location: ASR_model/openai-whisper-small/                      │
│  • Automatic download if not present                            │
│  • GPU acceleration (CUDA if available)                         │
│  • Returns transcribed text                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

**实时模式：**
```
麦克风 → PyAudio → 5秒分块 → Base64编码 → WebSocket(8765)
    → RabbitMQ → ASR模型 → 响应 → 写入 realtime_results.txt
    → 前端每2秒轮询结果
```

**批量模式：**
```
HuggingFace数据集 → 100个并发请求 → WebSocket(8765)
    → RabbitMQ → ASR模型 → 收集结果 → 返回前端
```

---

## 项目结构

```
useful-file/
├── ASR_model.py              # Whisper 模型封装
├── ASR_server.py             # RabbitMQ 消费者 + 模型处理器
├── ASR_websockets.py         # WebSocket 网关（RPC 桥接）
├── frontend_api.py           # FastAPI 服务器（进程管理）
├── client_real_mimic_api.py  # 实时音频客户端（PyAudio）
├── auto_dataset_client_mimic.py  # 批量测试客户端（datasets）
├── requirements.txt          # Python 依赖
├── log.py                    # 日志配置
├── frontend/
│   ├── index.html           # Web 界面
│   └── script.js            # 前端逻辑
├── ASR_model/
│   └── openai-whisper-small/  # 模型文件（自动下载）
└── log/                      # 应用日志
```

---

## 测试

### 命令行手动测试

**实时音频：**
```bash
python client_real_mimic_api.py
```

**批量测试：**
```bash
python auto_dataset_client_mimic.py
```

### Web 界面测试

1. 打开 `http://localhost:3006/frontend/index.html`
2. 点击 **"实时输入测试"** → 对着麦克风说话 → 点击 "停止"
3. 点击 **"多输入测试"** → 查看批量处理结果

---

## 故障排除

**问题：RabbitMQ 连接被拒绝**
```bash
sudo systemctl status rabbitmq-server
sudo systemctl start rabbitmq-server
```

**问题：未检测到麦克风（WSL）**
- PyAudio 在 WSL 上需要 PulseAudio
- 建议使用原生 Linux 或 Windows 以获得最佳麦克风支持

**问题：找不到模型**
- 首次运行会自动下载（可能需要 5-10 分钟）
- 检查网络连接
- 模型大小：约 500MB

**问题：CUDA 内存不足**
- 如果 CUDA 不可用，模型默认使用 CPU
- 检查 GPU 内存：`nvidia-smi`

---

## 配置

**端口：**
- 前端 API：`3006`
- WebSocket 网关：`8765`
- RabbitMQ：`5672`（默认）

**音频设置：**
- 采样率：`16kHz`
- 格式：`Float32`
- 块持续时间：`100ms`
- 发送间隔：`5 秒`

**代码修改：**
- `client_real_mimic_api.py`：修改 `RATE`、`SEND_INTERVAL`
- `ASR_model.py`：更改模型为 `openai-whisper-tiny` 以加快推理速度

---

## 许可证

教育/研究项目，用于 ASR 系统开发。

---

## 贡献者

AICARE 开发团队
