"""Genspark 网页端反代 —— 多账号轮转版

架构（标准网页端反代，浏览器不在链路里）：
  浏览器只用于登录取 cookie（gs_login.py）→ curl_cffi 纯 HTTP 转发

多账号轮转：
  读 accounts.json → 每个号一份 cookie + 独立 proxy
  轮转策略：round-robin + 失败自动切下一个号
  429/配额耗尽 → 冷却该号，切下一个

端点：
  POST /v1/chat/completions   (OpenAI 兼容，支持 stream)
  GET  /v1/models
  GET  /health
  GET  /state
"""
import json
import os
import re
import threading
import time
import uuid

from curl_cffi import requests as cffi
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BASE = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.environ.get("GS_ACCOUNTS", os.path.join(BASE, "accounts.json"))
PORT = int(os.environ.get("GS_PORT", "8899"))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
UPSTREAM = "https://www.genspark.ai/api/agent/ask_proxy"
REFERER = "https://www.genspark.ai/agents?type=ai_chat"

# 模型清单（2026-09-23 实测 53 个中 50 个可用）
MODELS = [
    # openai (23)
    "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6",
    "gpt-5-pro", "gpt-5.1-high", "gpt-5.1-low", "gpt-5.1-medium",
    "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro", "gpt-5.5-pro",
    "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-luna", "gpt-6-sol",
    # anthropic (10)
    "claude-4-5-haiku", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-opus-5", "claude-opus-5-5", "claude-sonnet-4", "claude-sonnet-4-5",
    "claude-sonnet-4-6", "claude-sonnet-5",
    # google (6)
    "gemini-2.5-flash", "gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview",
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    # genspark (11)
    "GLM-5.3", "deep-seek-v4-flash", "deep-seek-v4.1-flash", "glm-5p3",
    "glm-5p3-flash-baseten", "grok-4.5", "grok-4.6", "grok-4.7",
    "kimi-k3", "minimax-m3", "nemotron-3-ultra",
]
# 别名映射（用户可能用 API 风格的名字）
ALIAS = {
    "claude-haiku-4-5": "claude-4-5-haiku",
    "claude-opus-4-5": "claude-opus-4-6",
    "gpt-5.4-mini": "gpt-5.4-mini",
}

LOCK = threading.Lock()
_rr = 0


class Account:
    def __init__(self, d):
        self.seq = d.get("seq")
        self.email = d.get("email")
        self.cogen_id = d.get("cogen_id")
        cf = d.get("cookie_file")
        self.cookie_file = cf
        self.proxy = d.get("proxy") or d.get("proxy_default") or os.environ.get("GS_PROXY", "")
        self.cookie = ""
        self.cooldown_until = 0.0
        self.stats = {"ok": 0, "fail": 0, "throttle": 0}
        self.load()

    def load(self):
        if not self.cookie_file or not os.path.exists(self.cookie_file):
            self.cookie = ""
            return
        d = json.load(open(self.cookie_file, encoding="utf-8"))
        self.cookie = "; ".join(f"{c['name']}={c['value']}"
                                for c in d.get("cookies", []) if c.get("name"))

    @property
    def ready(self):
        return bool(self.cookie) and time.time() >= self.cooldown_until

    def cooldown(self, secs):
        self.cooldown_until = time.time() + secs

    def headers(self):
        rid = "|" + uuid.uuid4().hex + "." + uuid.uuid4().hex[:16]
        p = rid.lstrip("|").split(".")
        return {
            "User-Agent": UA, "Content-Type": "application/json",
            "Accept": "text/event-stream", "Origin": "https://www.genspark.ai",
            "Referer": REFERER, "request-id": rid,
            "traceparent": f"00-{p[0]}-{p[1]}-01", "Cookie": self.cookie,
        }

    @property
    def proxies(self):
        return {"https": self.proxy, "http": self.proxy}


def load_accounts():
    if not os.path.exists(MAP_FILE):
        raise RuntimeError(f"缺少 {MAP_FILE}")
    d = json.load(open(MAP_FILE, encoding="utf-8"))
    accts = []
    for a in d.get("accounts", []):
        if a.get("status") == "disabled":
            continue
        acc = Account(a)
        if acc.cookie:
            accts.append(acc)
    return accts


ACCOUNTS = load_accounts()
print(f"[init] 加载 {len(ACCOUNTS)} 个账号: "
      f"{[(a.seq, a.email[:22]) for a in ACCOUNTS]}", flush=True)


def pick():
    """round-robin 选可用账号"""
    global _rr
    with LOCK:
        ready = [a for a in ACCOUNTS if a.ready]
        if not ready:
            return None
        a = ready[_rr % len(ready)]
        _rr += 1
        return a


# ---------------------------------------------------------------- tool emulation
# The upstream web session does NOT accept OpenAI-style `tools`. Measured
# 2026-09-23: the parameter is accepted (HTTP 200) but ignored, and every model
# answers "I can't call a tool here" (verified on gpt-6-luna, claude-opus-5-5,
# gemini-3.8-flash, GLM-5.3).
#
# So tools are emulated at the gateway, the usual approach for web bridges:
#   1. render the tool schemas into a system prompt with a strict output contract
#   2. parse the model's reply back into OpenAI `tool_calls`
#   3. flatten tool-protocol messages the upstream rejects (HTTP 422)
#
# Verified prerequisites: the upstream honours role=system, and the model
# follows the output contract exactly (3/3 cases, including correctly NOT
# calling a tool when none applies).

TOOL_CONTRACT = """You have access to the following tools.

To call a tool, reply with EXACTLY one line of JSON and nothing else:
{"tool_call": {"name": "<tool_name>", "arguments": {<arguments>}}}

If no tool is needed, reply normally in plain text. Never emit the JSON line
unless you actually need a tool.

Available tools:
%s

Rules:
- Emit ONLY the JSON line when calling a tool: no prose, no markdown fences.
- "arguments" must be a valid JSON object matching that tool's parameters.
- One tool call per reply. If several are needed, call the first one now; the
  remaining ones will be requested after its result comes back."""


def render_tools(tools):
    """Render OpenAI tool schemas into the contract prompt."""
    lines = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not fn:
            fn = t if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        params = fn.get("parameters") or {}
        lines.append(f"- {name}: {desc}\n  arguments schema: "
                     f"{json.dumps(params, ensure_ascii=False)}")
    return TOOL_CONTRACT % ("\n".join(lines) if lines else "(none)")


def inject_tools(messages, tools):
    """Prepend the tool contract as a system message."""
    prompt = render_tools(tools)
    out = list(messages or [])
    if out and out[0].get("role") == "system":
        out[0] = {"role": "system",
                  "content": prompt + "\n\n" + str(out[0].get("content") or "")}
    else:
        out.insert(0, {"role": "system", "content": prompt})
    return out


def normalize_tool_messages(messages):
    """Translate OpenAI tool-protocol messages into plain chat messages.

    The upstream rejects the native shapes with HTTP 422 (measured 2026-09-23):
    an assistant message carrying `tool_calls`, or a message with role="tool".
    Both must be flattened:

      assistant{tool_calls:[...]}  -> assistant{content: <contract JSON line>}
      tool{tool_call_id, content}  -> user{content: "TOOL RESULT ..."}

    Verified: the flattened form round-trips and the model uses the result.
    """
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")

        if role == "tool":
            name = m.get("name") or "tool"
            cid = m.get("tool_call_id") or ""
            body = m.get("content")
            if not isinstance(body, str):
                body = json.dumps(body, ensure_ascii=False)
            head = f"TOOL RESULT for {name}" + (f" (call id {cid})" if cid else "")
            out.append({"role": "user",
                        "content": f"{head}: {body}\n"
                                   "Use this result to answer the user's question."})
            continue

        if role == "assistant" and m.get("tool_calls"):
            rendered = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                rendered.append(json.dumps(
                    {"tool_call": {"name": fn.get("name"), "arguments": args}},
                    ensure_ascii=False))
            content = "\n".join(rendered)
            if m.get("content"):
                content = str(m["content"]) + "\n" + content
            out.append({"role": "assistant", "content": content})
            continue

        clean = {"role": role, "content": m.get("content")}
        if m.get("name") and role != "assistant":
            clean["name"] = m["name"]
        out.append(clean)
    return out


def _extract_braced(s, start):
    """Return the substring from `start` (a '{') through its matching '}'.

    A non-greedy regex is wrong here: the payload nests objects
    ({"name":..., "arguments":{...}}), so `\\{.*?\\}` stops at the first inner
    brace and yields truncated JSON. Track depth, and respect string literals
    so a brace inside a string does not throw off the count.

    If the string ends before the depth returns to zero, the model dropped its
    closing brace(s) -- observed in practice, e.g. 65 chars where 66 were
    expected. Close them so the payload still parses.
    """
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    # unterminated: close the open object(s) and retry. A dangling string is
    # NOT repaired -- closing it would invent content the model never produced.
    if depth > 0 and not in_str:
        return s[start:] + "}" * depth
    return None


TOOLCALL_KEY_RE = re.compile(r'\{\s*"tool_call"\s*:\s*\{')


def parse_toolcall(text):
    """Extract a tool call from the model's reply.

    Returns (name, arguments_dict) or (None, None). Tolerates markdown fences
    and surrounding prose, since models sometimes add either despite the
    contract.
    """
    if not text:
        return None, None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()

    m = TOOLCALL_KEY_RE.search(s)
    if not m:
        return None, None
    outer = _extract_braced(s, m.start())
    if not outer:
        return None, None
    try:
        obj = json.loads(outer)
    except Exception:
        return None, None
    inner = obj.get("tool_call")
    if not isinstance(inner, dict):
        return None, None
    name = inner.get("name")
    args = inner.get("arguments")
    if not isinstance(name, str) or not name:
        return None, None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        try:
            args = json.loads(args)
        except Exception:
            return None, None
        if not isinstance(args, dict):
            return None, None
    return name, args


def build_body(payload):
    m = payload.get("model") or "claude-4-5-haiku"
    m = ALIAS.get(m, m)
    msgs = payload.get("messages") or []
    # Always flatten OpenAI tool-protocol messages: the upstream rejects
    # assistant.tool_calls and role="tool" with HTTP 422.
    msgs = normalize_tool_messages(msgs)
    tools = payload.get("tools") or []
    if tools:
        msgs = inject_tools(msgs, tools)
    return {
        "ai_chat_model": m,
        "ai_chat_enable_search": False,
        "ai_chat_disable_personalization": False,
        "use_moa_proxy": False, "moa_models": [], "writingContent": None,
        "sas_ask_origin": "typed", "type": "ai_chat", "is_private": True,
        "messages": msgs,
    }


def is_upstream_error(text):
    """Detect the upstream's canned failure strings.

    The web session occasionally answers with a placeholder instead of a real
    reply, e.g. "Sorry, I couldn't produce a response this turn." Returning that
    as normal content misleads the caller, and it silently breaks tool emulation
    (no contract line is produced). Treat it as a failure so the caller retries
    on another account.
    """
    if not text:
        return False
    t = text.strip().lower()
    if len(t) > 300:
        return False
    return any(s in t for s in (
        "couldn't produce a response",
        "could not produce a response",
        "unable to produce a response",
        "please try again",
        "something went wrong",
        "an error occurred",
        "服务异常", "请稍后再试", "出了点问题",
    ))


def parse_sse(text):
    content, deltas, throttle, err = None, [], None, None
    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            j = json.loads(line[6:])
        except Exception:
            continue
        t = j.get("type")
        if t == "message_field" and j.get("field_name") == "content":
            content = j.get("field_value")
        if t == "message_field_delta" and j.get("field_name") == "content":
            deltas.append(j.get("delta") or "")
        if t == "message_result" and isinstance(j.get("message"), dict):
            mc = j["message"].get("content") or ""
            if "too quickly" in mc or "Rate limit" in mc or "积分已用完" in mc:
                throttle = mc[:200]
            elif not content:
                content = mc
        if t == "error":
            err = json.dumps(j)[:300]
    return content, "".join(deltas), throttle, err


app = FastAPI()
START = time.time()


@app.get("/health")
def health():
    return {
        "ok": True, "uptime_s": round(time.time() - START, 1),
        "accounts": [{
            "seq": a.seq, "email": a.email[:26],
            "ready": a.ready,
            "cooldown_left_s": max(0, round(a.cooldown_until - time.time())),
            "stats": a.stats,
        } for a in ACCOUNTS],
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": m, "object": "model", "owned_by": "genspark-web"} for m in MODELS]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    payload = await req.json()
    model = payload.get("model") or "claude-4-5-haiku"
    want_stream = bool(payload.get("stream"))
    body = build_body(payload)
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    # 尝试轮转（最多试 5 个号）
    # 上游偶发返回占位符（约 1/3 概率），需要留足重试余量
    last_err = None
    for attempt in range(5):
        acct = pick()
        if acct is None:
            return JSONResponse(
                {"error": {"message": "所有账号都在冷却中（配额耗尽）",
                           "type": "no_account"}}, status_code=429)
        s = cffi.Session(impersonate="chrome")

        if not want_stream:
            try:
                r = s.post(UPSTREAM, headers=acct.headers(),
                           data=json.dumps(body), proxies=acct.proxies, timeout=120)
                t = r.text
            except Exception as e:
                acct.stats["fail"] += 1
                acct.cooldown(30)
                last_err = f"{type(e).__name__}: {e}"
                continue

            if "not login" in t:
                acct.stats["fail"] += 1
                acct.cooldown(300)
                last_err = "not_login"
                continue
            if "Rate limit" in t or "too quickly" in t:
                acct.stats["throttle"] += 1
                acct.cooldown(3600)
                last_err = "rate_limit"
                continue

            content, joined, throttle, err = parse_sse(t)
            if throttle:
                acct.stats["throttle"] += 1
                acct.cooldown(3600)
                last_err = "throttled"
                continue
            full = content or joined or ""

            # The upstream sometimes answers with a canned failure placeholder.
            # Retry on another account rather than passing it off as content.
            if is_upstream_error(full):
                acct.stats["fail"] += 1
                acct.cooldown(60)
                last_err = f"upstream_placeholder: {full[:80]}"
                continue

            acct.stats["ok"] += 1
            msg = {"role": "assistant", "content": full}
            finish = "stop"

            # tool emulation: turn a parsed contract line into OpenAI tool_calls
            if payload.get("tools"):
                tname, targs = parse_toolcall(full)
                if tname:
                    msg["content"] = None
                    msg["tool_calls"] = [{
                        "id": "call_" + uuid.uuid4().hex[:24],
                        "type": "function",
                        "function": {"name": tname,
                                     "arguments": json.dumps(targs, ensure_ascii=False)},
                    }]
                    finish = "tool_calls"

            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created,
                "model": model,
                "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "x_genspark": {"account": acct.seq, "email": acct.email[:22],
                               "upstream_status": r.status_code, "raw_len": len(t),
                               "tool_emulated": bool(payload.get("tools"))},
            })

        # 流式
        # Retries internally: a placeholder reply arrives before anything is
        # emitted (tools are buffered; without tools we only retry while nothing
        # has been sent), so the account can still be switched.
        def gen(i=cid, cr=created, mo=model):
            has_tools = bool(payload.get("tools"))
            yield f'data: {json.dumps({"id": i, "object": "chat.completion.chunk", "created": cr, "model": mo, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})}\n\n'
            last = None
            for _ in range(5):
                a = pick()
                if a is None:
                    yield f'data: {json.dumps({"error": {"message": "所有账号都在冷却中（配额耗尽）"}})}\n\n'
                    return
                buf, emitted, collected = "", 0, ""
                try:
                    r = cffi.Session(impersonate="chrome").post(
                        UPSTREAM, headers=a.headers(), data=json.dumps(body),
                        proxies=a.proxies, timeout=120, stream=True)
                    for chunk in r.iter_content(chunk_size=None):
                        if not chunk:
                            continue
                        buf += chunk.decode("utf-8", "replace")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if not line.startswith("data: "):
                                continue
                            try:
                                j = json.loads(line[6:])
                            except Exception:
                                continue
                            if j.get("type") == "message_field_delta" and j.get("field_name") == "content":
                                d = j.get("delta") or ""
                                if d:
                                    collected += d
                                    # With tools we cannot retract content
                                    # already sent, so buffer and decide later.
                                    if not has_tools:
                                        emitted += 1
                                        yield f'data: {json.dumps({"id": i, "object": "chat.completion.chunk", "created": cr, "model": mo, "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}]})}\n\n'
                            elif j.get("type") == "message_result" and isinstance(j.get("message"), dict):
                                mc = j["message"].get("content") or ""
                                if ("too quickly" in mc or "Rate limit" in mc or "积分已用完" in mc) and emitted == 0 and not has_tools:
                                    yield f'data: {json.dumps({"error": {"message": mc[:200]}})}\n\n'
                except Exception as e:
                    last = f"{type(e).__name__}: {e}"
                    a.stats["fail"] += 1
                    a.cooldown(30)
                    if emitted == 0:
                        continue
                    yield f'data: {json.dumps({"error": {"message": last}})}\n\n'
                    return

                placeholder = is_upstream_error(collected)
                if has_tools:
                    tname, targs = parse_toolcall(collected)
                    if tname:
                        tcid = "call_" + uuid.uuid4().hex[:24]
                        head = {"index": 0, "delta": {"tool_calls": [{
                            "index": 0, "id": tcid, "type": "function",
                            "function": {"name": tname, "arguments": ""}}]}}
                        yield f'data: {json.dumps({"id": i, "object": "chat.completion.chunk", "created": cr, "model": mo, "choices": [head]})}\n\n'
                        argchunk = {"index": 0, "delta": {"tool_calls": [{
                            "index": 0,
                            "function": {"arguments": json.dumps(targs, ensure_ascii=False)}}]}}
                        yield f'data: {json.dumps({"id": i, "object": "chat.completion.chunk", "created": cr, "model": mo, "choices": [argchunk]})}\n\n'
                        a.stats["ok"] += 1
                        fin = "tool_calls"
                    elif placeholder:
                        a.stats["fail"] += 1
                        a.cooldown(60)
                        last = f"upstream_placeholder: {collected[:80]}"
                        continue
                    else:
                        if collected:
                            yield f'data: {json.dumps({"id": i, "object": "chat.completion.chunk", "created": cr, "model": mo, "choices": [{"index": 0, "delta": {"content": collected}, "finish_reason": None}]})}\n\n'
                        a.stats["ok"] += 1
                        fin = "stop"
                else:
                    if placeholder and emitted == 0:
                        a.stats["fail"] += 1
                        a.cooldown(60)
                        last = f"upstream_placeholder: {collected[:80]}"
                        continue
                    a.stats["ok"] += 1
                    fin = "stop"

                yield f'data: {json.dumps({"id": i, "object": "chat.completion.chunk", "created": cr, "model": mo, "choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})}\n\n'
                yield "data: [DONE]\n\n"
                return

            yield f'data: {json.dumps({"error": {"message": f"所有账号都失败: {last}"}})}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return JSONResponse({"error": {"message": f"所有账号都失败: {last_err}"}},
                        status_code=502)


if __name__ == "__main__":
    import uvicorn
    print(f"[main] serving on :{PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
