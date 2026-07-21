# 环境搭建与踩坑记录（本地复现指南）

> 本文件记录了让本项目在新机器上跑通的完整步骤，以及调试过程中解决的兼容性问题。
> 原 `requirements.txt` 只锁了直接依赖，间接依赖会被 pip 装成最新版，导致大量不兼容。
> **推荐用 `requirements.lock.txt` 复现能跑的环境。**

## 一、前置要求

- **conda**（Miniconda 或 Anaconda）。原项目基于 Python 3.10，用系统新版 Python（如 3.14）会因为老包没有对应 wheel 而装不上。
- 两个 API Key：
  - **DeepSeek**（负责文本生成）
  - **SiliconFlow / 硅基流动**（负责 embedding，调用 `BAAI/bge-m3`）

## 二、搭建步骤

```bash
# 1. 建 Python 3.10 环境
conda create -n OpenPatent python=3.10 -y
conda activate OpenPatent

# 2. 安装锁定版依赖（重要：用 lock 文件，不要用 requirements.txt）
pip install -r requirements.lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 打补丁（见下方"三、必打的补丁"，重装环境后必须重做）

# 4. 在项目根目录建 .env 文件（见下方"四、.env 配置"）

# 5. 启动
python src/web_ui.py
# 浏览器打开 http://127.0.0.1:7860
```

## 三、必打的补丁（重装环境后必须重做！）

**问题**：`gradio_client 1.3.0` 在生成 API schema 时遇到布尔类型会崩溃
（`TypeError: argument of type 'bool' is not iterable`），导致首页 500、服务无法启动。

**修复**：编辑虚拟环境里的库文件
`<conda环境路径>/lib/site-packages/gradio_client/utils.py`

找到函数 `_json_schema_to_python_type`（约第 897 行），把：

```python
def _json_schema_to_python_type(schema: Any, defs) -> str:
    """Convert the json schema into a python type hint"""
    if schema == {}:
        return "Any"
```

改成（只在判断里增加 `or isinstance(schema, bool)`）：

```python
def _json_schema_to_python_type(schema: Any, defs) -> str:
    """Convert the json schema into a python type hint"""
    if schema == {} or isinstance(schema, bool):
        return "Any"
```

> 注意：`pip install` 重装会覆盖这个补丁，需要重新改一次。

## 四、.env 配置

在项目根目录（与 `src/` 同级）创建 `.env` 文件：

```
# === LLM 配置（换 API 只改这三行）===
LLM_API_BASE=https://api.deepseek.com
LLM_API_KEY=你的deepseek密钥
LLM_MODEL=deepseek-chat

# === Embedding 配置（向量库用，一般不动）===
SILICONFLOW_API_KEY=你的siliconflow密钥
```

以后切换到 GPT 等其他兼容 OpenAI 接口的服务，只改前三行即可：
```
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=你的openai密钥
LLM_MODEL=gpt-4o
```

## 五、本次调通过程中改过的代码

| 文件 | 改动 |
|------|------|
| `src/llm_integration.py` | `api_base`/`api_key`/`model` 改为从环境变量读取；两处 `model=` 用 `self.model` |
| `src/web_ui.py` | 删除 `gr.routes.client = AsyncClient(verify=False)` 及其 import；注释掉写死的 `SSL_CERT_FILE` 路径；`launch()` 加 `show_api=False` |

## 六、关键依赖版本（兼容组合，勿随意升级）

| 包 | 版本 | 说明 |
|----|------|------|
| gradio | 4.44.1 | 核心 UI |
| gradio_client | 1.3.0 | 需打补丁（见三） |
| fastapi | 0.112.2 | 必须配 gradio 4.44，勿升 |
| starlette | 0.38.6 | 必须配 gradio 4.44，勿升 |
| huggingface-hub | 0.20.3 | 高版本删了 `HfFolder`，勿升 |
</content>
</invoke>
