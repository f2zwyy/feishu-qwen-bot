import os
import json
import logging
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 环境变量（在 ECS 上设置） ----------
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

app = FastAPI(title="Feishu-Qwen-Bot")

# ---------- 通义千问 ----------
def call_qwen(user_msg: str) -> str:
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
        "parameters": {
            "result_format": "message"
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        output = data.get("output", {})
        choices = output.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return content

        text = output.get("text")
        if text:
            return text

        return "通义千问没有返回内容，请稍后重试。"
    except Exception as e:
        logger.exception("调用通义千问失败")
        return f"调用通义千问出错：{e}"


# ---------- 飞书数据模型（不再使用 schema 字段） ----------
class FeishuEventEnvelope(BaseModel):
    header: Optional[Dict[str, Any]] = None
    event: Optional[Dict[str, Any]] = None
    challenge: Optional[str] = None
    type: Optional[str] = None
    token: Optional[str] = None


# ---------- 获取 tenant_access_token，用来回消息 ----------
_tenant_access_token: Optional[str] = None
_tenant_access_token_expire: int = 0  # 时间戳，简单实现

def get_tenant_access_token() -> str:
    """
    根据 app_id / app_secret 获取 tenant_access_token
    """
    import time

    global _tenant_access_token, _tenant_access_token_expire

    now = int(time.time())
    if _tenant_access_token and now < _tenant_access_token_expire - 60:
        return _tenant_access_token

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = {
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET,
    }
    resp = requests.post(url, json=payload, timeout=10)
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"获取 tenant_access_token 失败: {data}")

    _tenant_access_token = data["tenant_access_token"]
    _tenant_access_token_expire = now + data.get("expire", 3600)
    return _tenant_access_token


def feishu_reply_message(message_id: str, text: str) -> None:
    """
    调用飞书接口，回复一条消息。
    这里用的是「reply」接口，你也可以改成按 chat_id 发新消息。
    """
    if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
        logger.error("FEISHU_APP_ID / FEISHU_APP_SECRET 未配置，无法回复消息")
        return

    try:
        token = get_tenant_access_token()
    except Exception:
        logger.exception("获取 tenant_access_token 失败，无法回复消息")
        return

    url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "msg_type": "text",
        "content": json.dumps({"text": text}, ensure_ascii=False),
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        data = resp.json()
        if data.get("code") != 0:
            logger.error(f"回复消息失败: {data}")
        else:
            logger.info("已回复飞书消息")
    except Exception:
        logger.exception("调用飞书回复接口失败")


# ---------- 健康检查 ----------
@app.get("/")
async def root():
    return {"message": "Feishu-Qwen-Bot is running."}


# ---------- 飞书回调入口 ----------
@app.post("/feishu/webhook")
async def feishu_webhook(request: Request):
    body = await request.json()
    logger.info(f"收到飞书请求: {body}")

    envelope = FeishuEventEnvelope(**body)

    # 1) URL 校验
    if envelope.type == "url_verification" and envelope.challenge:
        return {"challenge": envelope.challenge}

    # 2) 事件回调
    event = envelope.event or {}
    event_type = event.get("type")

    if event_type == "im.message.receive_v1":
        message = event.get("message", {})
        msg_type = message.get("message_type")
        content_raw = message.get("content", "{}")
        message_id = message.get("message_id", "")

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

        # 主动调用飞书 API 回复
        if message_id:
            feishu_reply_message(message_id, reply_text)
        else:
            logger.warning("没有拿到 message_id，无法直接回复消息")

        return {"code": 0, "message": "ok"}

    # 其他事件先忽略
    return {"code": 0, "message": "ignored"}


# ---------- 直接 python3 app.py 运行 ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
