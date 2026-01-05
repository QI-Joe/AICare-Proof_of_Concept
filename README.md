# AICARE - 概念验证实验分支

> **⚠️ 实验性分支说明**  
> 本分支是一个**概念验证（Proof of Concept）实验分支**，主要目标是将原有的 **WebSocket 架构替换为 RabbitMQ + aio-pika** 异步消息队列架构，验证基于 RPC 模式的 ASR 系统的可行性和性能表现。

## 项目简介

AICARE 是一个基于 **OpenAI Whisper** 模型的自动语音识别（ASR）系统实验项目。本分支采用 **RabbitMQ 消息队列 + RPC 模式**重构了原有架构，实现：

- ✅ **异步消息队列架构**：完全基于 RabbitMQ + aio-pika 的异步 RPC 通信
- ✅ **Worker-Client 解耦**：Worker 独立处理 ASR/LLM 任务，Client 通过 RPC 调用服务
- ✅ **批量并发处理**：支持大规模并发音频样本的批量识别测试
- ✅ **系统稳定性验证**：提供连接心跳测试和压力测试工具
- ✅ **可扩展架构**：支持后续扩展 LLM 队列和其他 AI 模型服务

**核心技术栈**：Python, RabbitMQ, aio-pika, asyncio, OpenAI Whisper, PyTorch, Transformers

---

## 系统架构

### 核心架构图

```
┌──────────────────────────────────────────────────────────────┐
│                     客户端层 (Client Layer)                    │
├──────────────────────────────────────────────────────────────┤
│  • auto_dataset_client_mimic.py  - 批量测试客户端             │
│  • stability_test.py             - 稳定性测试工具             │
│  • client_real_mimic_api.py      - 实时音频客户端（可选）      │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      │ RPC_client.py (RPC 客户端封装)
                      │ • call_asr(audio, sample_rate)
                      │ • call_llm(text)
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│               消息队列层 (RabbitMQ Broker)                     │
│                    localhost:5672                             │
├──────────────────────────────────────────────────────────────┤
│  队列 1: asr_queue  (ASR 语音识别任务队列)                     │
│  队列 2: llm_queue  (LLM 文本处理任务队列 - 预留)              │
│                                                               │
│  模式: RPC (correlation_id + reply_to + callback_queue)       │
│  QoS: prefetch_count=2, durable=True                          │
│  心跳: heartbeat=60s                                          │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      │ aio_pika.connect_robust()
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                   工作节点层 (Worker Layer)                    │
│                       Worker.py                               │
├──────────────────────────────────────────────────────────────┤
│  • 初始化连接: initial_connection()                            │
│  • 声明队列: asr_queue, llm_queue                             │
│  • 消息处理:                                                   │
│    - asr_message_process() → 解码音频 → 调用 ASR 模型         │
│    - llm_message_process() → 处理文本（预留）                  │
│  • 异步消费: on_message() → 自动 ACK                           │
│  • 结果返回: 发布到 reply_to 队列                              │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      │ ASR_model.py
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                   模型层 (Model Layer)                         │
│                    ASR_model.py                               │
├──────────────────────────────────────────────────────────────┤
│  • 模型: openai/whisper-small                                 │
│  • 存储路径: ASR_model/openai-whisper-small/                  │
│  • 自动下载: 首次运行自动从 HuggingFace 下载                   │
│  • GPU 加速: 自动检测 CUDA，回退到 CPU                         │
│  • 处理流程: Base64 解码 → np.float32 → Whisper → 文本        │
└──────────────────────────────────────────────────────────────┘
```

### RPC 消息流

```
Client                    RabbitMQ                    Worker
  │                          │                          │
  │ 1. 生成 correlation_id    │                          │
  │ 2. 准备音频数据            │                          │
  │    (Base64 + JSON)       │                          │
  │                          │                          │
  ├─ publish ──────────────> │                          │
  │  (asr_queue)             │                          │
  │  reply_to=callback_queue │                          │
  │                          │                          │
  │                          ├─ consume ─────────────> │
  │                          │                          │
  │                          │  3. 解码音频              │
  │                          │  4. ASR 模型推理          │
  │                          │  5. 生成结果 JSON         │
  │                          │                          │
  │                          │ <── publish ─────────── │
  │                          │  (callback_queue)        │
  │                          │  correlation_id 匹配     │
  │                          │                          │
  │ <─ consume ────────────  │                          │
  │  6. Future.set_result()  │                          │
  │  7. 返回识别文本          │                          │
  │                          │                          │
```

---

## 核心代码说明

### 1. Worker.py - 核心工作节点

**功能**：消息队列的消费者，负责接收任务、调用模型、返回结果

```python
# 主要职责：
• initial_connection()    - 建立 RabbitMQ 连接，声明 asr_queue 和 llm_queue
• asr_message_process()   - 解码 Base64 音频，调用 ASR_model.process()
• llm_message_process()   - 预留的 LLM 文本处理接口
• on_message()            - 异步消息处理器，自动 ACK，通过 reply_to 返回结果
• run()                   - 启动多队列消费循环
```

**启动方式**：
```bash
python Worker.py
```

### 2. RPC_client.py - RPC 客户端封装

**功能**：封装 RabbitMQ RPC 调用逻辑，提供简洁的异步接口

```python
# 主要接口：
• connect()               - 建立连接，创建独占的 callback_queue
• call_asr(audio, sr)     - 发送音频到 asr_queue，等待结果（支持超时）
• call_llm(text)          - 发送文本到 llm_queue（预留接口）
• _on_response()          - 监听回调队列，根据 correlation_id 匹配 Future
```

**使用示例**：
```python
client = RPClient()
await client.connect()
result = await client.call_asr(audio_array, 16000)
print(result['text'])
await client.close()
```

### 3. auto_dataset_client_mimic.py - 批量测试客户端

**功能**：从 HuggingFace 数据集加载音频样本，并发测试 ASR 系统性能

```python
# 主要功能：
• single_request_test()          - 单个请求测试
• concurrent_requests_test()     - 并发请求测试（默认 10 个）
• run_batch_test_for_api()       - API 兼容的批量测试接口
```

**运行方式**：
```bash
python auto_dataset_client_mimic.py
```

**测试数据集**：`hf-internal-testing/librispeech_asr_dummy`（清洁语音样本）

### 4. stability_test.py - 系统稳定性测试工具

**功能**：验证系统的连接稳定性和高并发处理能力

#### 测试 1：连接心跳测试
```python
• 持续时间：60 分钟（可配置）
• 心跳间隔：每 10 秒检查一次连接状态
• 验证内容：RabbitMQ 连接是否保持活跃（heartbeat=60s）
```

#### 测试 2：请求压力测试
```python
• 请求数量：100 次（可配置）
• 并发数：10 个并发请求
• 测试音频：随机生成的 1 秒音频（16kHz, float32）
• 统计指标：成功率、平均响应时间、失败原因
```

**运行方式**：
```bash
python stability_test.py

# 选择测试：
# 1. 连接心跳测试（1 小时）
# 2. 请求压力测试（100 次）
# 3. 同时运行两个测试
```

### 5. ASR_model.py - Whisper 模型封装

**功能**：加载和管理 Whisper 模型，提供音频处理接口

```python
# 核心功能：
• load_model_test()           - 自动下载/加载模型，GPU 自动检测
• ASRModel.process()          - 音频数组 → 文本转录
• check_model_status()        - 输出模型设备、参数量、数据类型
```

**模型自动下载**：首次运行时自动从 HuggingFace 下载约 500MB 的 Whisper-small 模型

---

## 快速开始

### 前置要求

#### 系统依赖（Ubuntu/Debian）
```bash
sudo apt update
sudo apt install -y erlang rabbitmq-server
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
```

#### 系统依赖（macOS）
```bash
brew install rabbitmq
brew services start rabbitmq
```

### 安装步骤

**1. 激活虚拟环境（如果已存在）：**
```bash
cd /Users/qsh/Documents/AICare-Proof_of_Concept
source venv310/bin/activate
```

**2. 安装依赖：**
```bash
pip install -r requirements.txt
```

**3. 启动系统：**

```bash
# 终端 1：启动 Worker（必须）
python Worker.py
# 输出：✅ ASR queue consuming / ✅ All queues ready
```

### 运行测试

#### 批量测试（推荐）
```bash
# 终端 2：运行批量测试
python auto_dataset_client_mimic.py

# 输出示例：
# === Concurrent Requests Test (10 samples) ===
# [Client 1272-135031] Response received (elapsed: 2.34s): MR QUILTER...
# 10 tasks completed
```

#### 稳定性测试
```bash
# 终端 2：运行稳定性测试
python stability_test.py

# 选项：
# 1 - 连接心跳测试（持续 1 小时，每 10 秒检查）
# 2 - 压力测试（100 次请求，10 并发）
# 3 - 同时运行两个测试
```

---

## 测试结果示例

### 批量测试输出
```
=== Concurrent Requests Test (10 samples) ===
Ground truths:
  0: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES...
  1: NOR IS MISTER QUILTER'S MANNER LESS INTERESTING...

[Client 1272-135031] Response received (elapsed: 2.34s): MISTER QUILTER...
[Client 1272-135031-1] Response received (elapsed: 2.56s): NOR IS MISTER...
...
10 tasks completed
```

### 稳定性测试输出
```
[2026-01-05 10:30:00] Iteration 1: Connection healthy ✓
[2026-01-05 10:30:10] Iteration 2: Connection healthy ✓
...

Stress Test Progress: 50/100 (48 successful)
Test completed: 98/100 successful (98.0%)
```

---

## 项目结构

```
AICare-Proof_of_Concept/
├── Worker.py                        # 🔥 核心工作节点（消费者）
├── RPC_client.py                    # 🔥 RPC 客户端封装
├── ASR_model.py                     # 🔥 Whisper 模型封装
│
├── auto_dataset_client_mimic.py     # 📊 批量测试客户端
├── stability_test.py                # 📊 稳定性测试工具
├── client_real_mimic_api.py         # （可选）实时音频客户端
│
├── frontend_api.py                  # （可选）Web API 服务器
├── log.py                           # 日志配置
├── requirements.txt                 # Python 依赖
│
├── ASR_model/
│   └── openai-whisper-small/        # Whisper 模型文件（自动下载）
├── frontend/                        # （可选）Web 前端
├── log/                             # 日志文件目录
└── venv310/                         # Python 虚拟环境
```

---

## 配置说明

### RabbitMQ 配置
- **连接地址**：`localhost:5672`
- **用户名/密码**：`guest/guest`
- **心跳间隔**：`60 秒`
- **队列持久化**：`durable=True`
- **QoS 预取**：`prefetch_count=2`

### 音频处理配置
- **采样率**：`16kHz`
- **数据类型**：`np.float32`
- **编码方式**：`Base64`

### 模型配置
- **模型名称**：`openai/whisper-small`
- **存储路径**：`ASR_model/openai-whisper-small/`
- **设备选择**：自动检测 CUDA，回退到 CPU

---

## 故障排除

### RabbitMQ 连接失败
```bash
# 检查 RabbitMQ 状态
sudo systemctl status rabbitmq-server

# 启动 RabbitMQ
sudo systemctl start rabbitmq-server

# 查看日志
sudo journalctl -u rabbitmq-server -f
```

### Worker 无法启动
```bash
# 检查端口占用
lsof -i :5672

# 清空队列（如果需要）
# 进入 RabbitMQ 管理界面：http://localhost:15672
# 用户名/密码：guest/guest
```

### 模型下载失败
- 检查网络连接
- 尝试手动下载：`huggingface-cli download openai/whisper-small`
- 或使用镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 实验结果与结论

本实验分支成功验证了：

✅ **RabbitMQ + aio-pika** 架构可以完全替代 WebSocket，实现更松耦合的异步通信  
✅ **RPC 模式**能够有效处理请求-响应模式的 ASR 任务  
✅ **并发性能**良好，支持 10+ 并发请求的稳定处理  
✅ **连接稳定性**可靠，心跳机制确保长时间运行不断连  
✅ **可扩展性**强，轻松扩展 LLM 队列和其他 AI 服务

---

## 后续计划

- [ ] 集成 LLM 模型处理管道
- [ ] 实现分布式 Worker 负载均衡
- [ ] 添加任务优先级队列
- [ ] 实现结果持久化存储
- [ ] 性能监控和指标收集

---

## 开发团队

AICARE 实验团队 - 2026
