# app.py
# 简化版：不使用加密、不校验签名，只处理明文事件（适合入门和调试）
# 依赖: flask, requests

import os
import time
import json

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# ========= 已处理事件缓存（用于去重） =========
# key: event_id, value: 处理时间戳
PROCESSED_EVENTS = {}
PROCESSED_TTL = 60 * 5  # 只保存最近 5 分钟的 event_id，防止内存无限增长

# ========= 环境变量 =========
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

_access_token = None
_token_expire = 0


# ========= 工具函数 =========

def get_tenant_access_token():
    """获取 / 缓存 tenant_access_token"""
    global _access_token, _token_expire

    if _access_token and time.time() < _token_expire:
        return _access_token

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    resp = requests.post(url, json={
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET
    }, timeout=10)
    data = resp.json()
    if resp.status_code != 200 or "tenant_access_token" not in data:
        print("❌ 获取 tenant_access_token 失败:", resp.status_code, data)
        return None

    _access_token = data["tenant_access_token"]
    _token_expire = time.time() + data.get("expire", 3600) - 60
    print("✅ 获取 tenant_access_token 成功")
    return _access_token


def call_qwen(prompt: str) -> str:
    """调用通义千问（DashScope）"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-turbo",
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if resp.status_code == 200:
            content = data["output"]["choices"][0]["message"]["content"]
            return content
        else:
            print("❌ Qwen 返回错误:", resp.status_code, data)
            return "通义千问调用失败，请稍后再试～"
    except Exception as e:
        print("❌ 调用通义千问异常:", e)
        return "通义千问接口异常，请稍后再试～"


def send_message(chat_id: str, text: str):
    """给指定 chat_id 发送文本消息"""
    token = get_tenant_access_token()
    if not token:
        print("❌ 无法获取 tenant_access_token，消息发送失败")
        return

    # 按照飞书文档，receive_id_type 放在 query 参数里
    url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    payload = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text}, ensure_ascii=False)
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    print("📨 发送消息结果:", resp.status_code, data)


def handle_event(event_data: dict):
    """处理飞书事件（只关心 im.message.receive_v1），带事件去重"""

    # 飞书新事件是 2.0 格式
    if event_data.get("schema") != "2.0":
        print("⚠️ 非 2.0 事件，直接忽略:", event_data)
        return

    header = event_data.get("header", {})
    event_type = header.get("event_type")
    event_id = header.get("event_id")  # 用于去重
    event = event_data.get("event", {})

    # ========= 去重逻辑开始 =========
    now = time.time()

    # 清理过期的 event_id
    expired_ids = [eid for eid, ts in PROCESSED_EVENTS.items() if now - ts > PROCESSED_TTL]
    for eid in expired_ids:
        PROCESSED_EVENTS.pop(eid, None)

    if event_id:
        if event_id in PROCESSED_EVENTS:
            print(f"♻️ 收到重复事件，event_id={event_id}，不再处理")
            return
        # 先记录为已处理，避免中途出错又重复处理
        PROCESSED_EVENTS[event_id] = now
    else:
        print("⚠️ 事件没有 event_id，无法去重")
    # ========= 去重逻辑结束 =========

    if event_type == "im.message.receive_v1":
        message = event.get("message", {})
        chat_id = message.get("chat_id")
        msg_type = message.get("message_type")  # text/image 等
        content_str = message.get("content", "{}")

        # content 是一个 JSON 字符串，如 {"text": "你好"}
        try:
            content_obj = json.loads(content_str)
        except Exception as e:
            print("❌ 解析 message.content 失败:", e, content_str)
            return

        user_text = content_obj.get("text", "").strip()

        print(f"💬 收到消息: event_id={event_id}, chat_id={chat_id}, type={msg_type}, text={user_text}")

        if chat_id and msg_type == "text" and user_text:
            reply = call_qwen(user_text)
            send_message(chat_id, reply)
    else:
        print("⚠️ 收到其它事件类型:", event_type)


# ========= 路由 =========

@app.route("/feishu/webhook", methods=["POST"])
def feishu_webhook():
    """
    飞书事件回调入口（无加密，无签名）
    1. URL 校验：type == url_verification
    2. 普通事件：schema == 2.0
    """
    raw_body = request.get_data(as_text=True) or ""
    print("👉 收到原始请求:", raw_body)

    try:
        data = json.loads(raw_body)
    except Exception as e:
        print("❌ JSON 解析失败:", e, raw_body)
        # 一定返回合法 JSON，避免飞书提示“非法 JSON”
        return jsonify({"code": 1, "msg": "bad json", "detail": str(e)}), 400

    # 1. URL 校验
    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        print("✅ URL 校验请求，challenge =", challenge)
        return jsonify({"challenge": challenge})

    # 2. 普通事件
    try:
        handle_event(data)
    except Exception as e:
        print("❌ 处理事件异常:", e)
        # 返回 200 + JSON，避免飞书一直重试
        return jsonify({"code": 0, "msg": "event error", "detail": str(e)})

    return jsonify({"code": 0, "msg": "ok"})


@app.route("/")
def home():
    return jsonify({"status": "Feishu Qwen Bot is running (no-encrypt version, with dedupe)"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
