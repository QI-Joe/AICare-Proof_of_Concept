from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uuid
import subprocess
import os
import json

app = FastAPI()

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务存储
tasks = {}
real_time_process = None

class BatchTestRequest(BaseModel):
    num_samples: int = 10

# 静态文件服务
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def root():
    return {"message": "ASR Frontend API", "frontend_url": "/frontend/index.html"}

@app.post("/api/asr/real-time/start")
async def start_real_time():
    global real_time_process
    
    if real_time_process and real_time_process.poll() is None:
        return {"status": "already_running", "pid": real_time_process.pid}
    
    # 清空旧结果
    with open("realtime_results.txt", "w") as f:
        f.write("")
    
    # 启动进程
    real_time_process = subprocess.Popen(
        ["python", "client_real_mimic_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return {"status": "started", "pid": real_time_process.pid}

@app.post("/api/asr/real-time/stop")
async def stop_real_time():
    global real_time_process
    
    if real_time_process and real_time_process.poll() is None:
        real_time_process.terminate()
        real_time_process.wait(timeout=5)
        real_time_process = None
        return {"status": "stopped"}
    
    return {"status": "not_running"}

@app.get("/api/asr/real-time/status")
async def get_real_time_status():
    global real_time_process
    
    if real_time_process and real_time_process.poll() is None:
        return {"running": True, "pid": real_time_process.pid}
    
    return {"running": False}

@app.get("/api/asr/real-time/results")
async def get_real_time_results():
    try:
        with open("realtime_results.txt", "r") as f:
            lines = f.readlines()
            results = []
            for line in lines:
                if line.strip():
                    results.append(json.loads(line))
            return {"results": results}
    except FileNotFoundError:
        return {"results": []}

@app.post("/api/asr/batch-test")
async def start_batch_test(request: BatchTestRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "running", "progress": 0, "results": []}
    
    # 在后台运行批量测试
    asyncio.create_task(run_batch_test(task_id, request.num_samples))
    
    return {"task_id": task_id, "status": "started"}

@app.get("/api/asr/batch-test/{task_id}")
async def get_batch_test_status(task_id: str):
    if task_id not in tasks:
        return {"error": "Task not found"}
    return tasks[task_id]

async def run_batch_test(task_id: str, num_samples: int):
    """运行批量测试任务"""
    # 导入测试函数
    from auto_dataset_client_mimic import run_batch_test_for_api
    
    try:
        results = await run_batch_test_for_api(num_samples)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["results"] = results
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3006)

