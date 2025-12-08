import os
import json
import logging
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel

# -------- 日志设置 --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- 读取环境变量（请在 ECS 上 export） --------
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")  # 可改成你常用的模型

app = FastAPI(title="Feishu-Qwen-Bot on Aliyun ECS")

# -------- 通义千问调用封装（DashScope HTTP 接口） --------
def call_qwen(user_msg: str) -> str:
    """
    调用通义千问，输入一段用户消息，返回一段文字回复。
    如果你习惯用 DashScope SDK 或 OpenAI 兼容接口，也可以自己替换这里。
    """
    if not QWEN_API_KEY:
        return "后端未配置 QWEN_API_KEY，请联系管理员设置环境变量。"

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }
    payload = {
        "model": QWEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个企业内部智能助手，回答要简洁、专业、直接，默认用简体中文。",
                },
                {"role": "user", "content": user_msg},
            ]
        },
        # 参考官方文档，可以加参数：temperature、max_tokens、result_format 等
        "parameters": {
            "result_format": "message"  # 新版推荐，用 message 格式
        },
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # 兼容几种返回格式
        output = data.get("output", {})
        # 1) 新版 message 格式
        choices = output.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return content

        # 2) 旧版 text 字段
        text = output.get("text")
        if text:
            return text

        return "通义千问没有返回内容，请稍后重试。"
    except Exception as e:
        logger.exception("调用通义千问失败")
        return f"调用通义千问出错：{e}"


# -------- 飞书请求数据模型（简单封装，方便类型提示） --------
class FeishuEventEnvelope(BaseModel):
    schema: Optional[str] = None
    header: Optional[Dict[str, Any]] = None
    event: Optional[Dict[str, Any]] = None
    challenge: Optional[str] = None
    type: Optional[str] = None
    token: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Feishu-Qwen-Bot is running on Aliyun ECS."}


# -------- 飞书回调入口（路径可以按照你在飞书配置的来改） --------
@app.post("/feishu/webhook")
async def feishu_webhook(request: Request):
    """
    飞书事件订阅 / 机器人回调入口。
    兼容：
    - URL 校验：type = url_verification
    - 事件回调：type = event_callback（消息事件等）
    """
    body = await request.json()
    logger.info(f"收到飞书请求: {body}")

    envelope = FeishuEventEnvelope(**body)

    # 1) URL 校验（飞书配置回调地址时第一次会发这种请求）
    if envelope.type == "url_verification" and envelope.challenge:
        return {"challenge": envelope.challenge}

    # 2) 普通事件回调
    event = envelope.event or {}
    event_type = event.get("type")

    # 这里只做最常见的“接收文本消息 -> 调用通义千问 -> 返回文本”
    # 你可以根据自己原来的逻辑，把 event 结构补全 / 修改。
    if event_type == "im.message.receive_v1":
        message = event.get("message", {})
        msg_type = message.get("message_type")
        content_raw = message.get("content", "{}")

        user_text = ""
        if msg_type == "text":
            try:
                content_json = json.loads(content_raw)
                user_text = content_json.get("text", "")
            except Exception:
                user_text = content_raw

        if not user_text:
            reply_text = "我收到了一个空消息，能再发一遍吗？"
        else:
            reply_text = call_qwen(user_text)

        # ⚠️ 注意：对“事件订阅机器人”，通常需要主动调用飞书接口回复消息，
        # 而不是直接在回调返回内容。
        # 你原来的项目如果已经实现了“主动调用飞书 API 回复”的逻辑，
        # 可以把那部分代码搬到这里来调用。
        #
        # 这里为了简单，回调只返回 200 + "ok"。
        # 真正的回复由你在现有代码里调用飞书 OpenAPI 完成。
        logger.info(f"准备回复用户: {reply_text}")

        return {"code": 0, "message": "ok"}

    # 其他事件类型暂时不处理
    return {"code": 0, "message": "ignored"}
