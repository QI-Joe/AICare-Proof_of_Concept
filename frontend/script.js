// API 基础地址
const API_BASE = 'http://localhost:3006';

// 获取 DOM 元素
const btnRealTime = document.getElementById('btn-real-time');
const btnBatchTest = document.getElementById('btn-batch-test');
const areaRealTime = document.getElementById('area-real-time');
const areaBatchTest = document.getElementById('area-batch-test');

// 按钮 A: 实时语音测试
let realTimeRunning = false;
let pollInterval = null;

btnRealTime.addEventListener('click', async () => {
    if (!realTimeRunning) {
        // 启动
        areaRealTime.innerHTML = '<h3>实时语音识别结果</h3><p class="status">正在启动...</p>';
        
        const response = await fetch(`${API_BASE}/api/asr/real-time/start`, {method: 'POST'});
        const data = await response.json();
        
        if (data.status === 'started') {
            realTimeRunning = true;
            btnRealTime.textContent = '实时语音测试 (停止)';
            areaRealTime.innerHTML = '<h3>实时语音识别结果</h3><p class="status">采集中...请对着麦克风说话</p>';
            
            // 轮询结果
            pollInterval = setInterval(async () => {
                const res = await fetch(`${API_BASE}/api/asr/real-time/results`);
                const resultData = await res.json();
                
                if (resultData.results.length > 0) {
                    let html = '<h3>实时语音识别结果</h3>';
                    resultData.results.forEach((item) => {
                        const time = new Date(item.timestamp * 1000).toLocaleTimeString();
                        html += `<div class="result-item">
                            <strong>${time}</strong><br>
                            ${item.text}
                        </div>`;
                    });
                    areaRealTime.innerHTML = html;
                    areaRealTime.scrollTop = areaRealTime.scrollHeight;
                }
            }, 2000);
        }
        
    } else {
        // 停止
        await fetch(`${API_BASE}/api/asr/real-time/stop`, {method: 'POST'});
        realTimeRunning = false;
        btnRealTime.textContent = 'Real Input Test';
        clearInterval(pollInterval);
        
        // 显示最终结果
        const res = await fetch(`${API_BASE}/api/asr/real-time/results`);
        const data = await res.json();
        let html = '<h3>实时语音识别结果 (已停止)</h3>';
        data.results.forEach((item) => {
            const time = new Date(item.timestamp * 1000).toLocaleTimeString();
            html += `<div class="result-item">
                <strong>${time}</strong><br>
                ${item.text}
            </div>`;
        });
        areaRealTime.innerHTML = html;
    }
});

// 按钮 B: 批量测试
btnBatchTest.addEventListener('click', async () => {
    areaBatchTest.innerHTML = '<h3>批量测试结果</h3><p class="status">启动中...</p>';
    
    const response = await fetch(`${API_BASE}/api/asr/batch-test`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({num_samples: 10})
    });
    const data = await response.json();
    
    const taskId = data.task_id;
    areaBatchTest.innerHTML = `<h3>批量测试结果</h3><p class="status">任务ID: ${taskId}<br>正在处理...</p>`;
    
    // 轮询获取结果
    const checkResult = async () => {
        const res = await fetch(`${API_BASE}/api/asr/batch-test/${taskId}`);
        const result = await res.json();
        
        if (result.status === 'completed') {
            let html = '<h3>批量测试结果</h3>';
            result.results.forEach((item, idx) => {
                html += `<div class="result-item">
                    <strong>#${idx + 1}</strong><br>
                    识别: ${item.recognized}<br>
                    真实: ${item.ground_truth}
                </div>`;
            });
            areaBatchTest.innerHTML = html;
        } else if (result.status === 'failed') {
            areaBatchTest.innerHTML = `<h3>批量测试结果</h3><p class="status" style="color: red;">❌ 测试失败<br><br>错误信息:<br>${result.error}</p>`;
        } else {
            setTimeout(checkResult, 2000);
        }
    };
    checkResult();
});
