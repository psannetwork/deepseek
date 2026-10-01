#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API Server - OpenAI-compatible server for DeepSeek
'''

import os
import json
import time
import sys
from io import StringIO
from flask import Flask, request, jsonify, Response, stream_with_context
from DeepSeekAPI import DeepSeekChat

app = Flask(__name__)

# Load tokens from file or environment
def get_tokens():
    if os.path.exists("tokens"):
        with open("tokens") as f:
            lines = f.read().strip().split('\n')
            if len(lines) >= 2:
                return lines[0], lines[1]
    ds_session_id = os.environ.get("DS_SESSION_ID")
    authorization_token = os.environ.get("AUTHORIZATION_TOKEN")
    if ds_session_id and authorization_token:
        return ds_session_id, authorization_token
    raise ValueError("Tokens not found. Set DS_SESSION_ID and AUTHORIZATION_TOKEN or create 'tokens' file.")

DS_SESSION_ID, AUTHORIZATION_TOKEN = get_tokens()

# Dictionary to store chat instances per session
chat_sessions = {}

def get_chat_instance(session_id="default"):
    if session_id not in chat_sessions:
        print(f"DEBUG: Creating new chat instance for session: {session_id}")
        chat_sessions[session_id] = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    else:
        print(f"DEBUG: Reusing existing chat instance for session: {session_id}")
    
    instance = chat_sessions[session_id]
    # Ensure session is created if missing
    if not instance.chat_session_id:
        print(f"DEBUG: Session ID missing for {session_id}, creating...")
        instance.create_chat_session()
        
    return instance

import re

def get_model_config(model: str):
    """Map OpenAI-compatible model names to DeepSeek web model flags."""
    model_lower = model.lower()
    model_type = "expert" if "v4" in model_lower or "r4" in model_lower or "expert" in model_lower else "default"
    thinking_enabled = "r1" in model_lower or "r4" in model_lower or "reasoning" in model_lower or "reasoner" in model_lower
    return model_type, thinking_enabled

def clean_response(text: str) -> str:
    if not text:
        return ""
    # Remove hidden HTML divs like <div style="display:none">...</div>
    text = re.sub(r'<div[^>]*>.*?</div>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Strip trailing FINISHED (even if repeated like FINISHEDFINISHED)
    text = re.sub(r'(?:FINISHED\s*)+$', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def format_tools_prompt(tools):
    if not tools:
        return ""
    tool_descriptions = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "")
        desc = func.get("description", "")
        params = json.dumps(func.get("parameters", {}), ensure_ascii=False)
        tool_descriptions.append(f"- Tool: {name}\n  Description: {desc}\n  Parameters: {params}")
    
    return "Available Tools:\n" + "\n".join(tool_descriptions) + "\n\nWhen you need to use a tool, output it in JSON format inside <tool_calls> tags, e.g.:\n<tool_calls>\n[{\"name\": \"tool_name\", \"arguments\": {...}}]\n</tool_calls>\n"

def format_messages(messages, tools=None):
    """Format OpenAI messages list into a single prompt string for DeepSeek, including system prompts, tools, conversation history, and tool responses."""
    if not messages and not tools:
        return ""
    
    system_parts = []
    chat_parts = []
    
    if tools:
        system_parts.append(format_tools_prompt(tools))
        
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            chat_parts.append(f"User: {content}")
        elif role == "assistant":
            if "tool_calls" in msg and msg["tool_calls"]:
                calls = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except:
                        args = fn.get("arguments", {})
                    calls.append({"name": fn.get("name"), "arguments": args})
                chat_parts.append(f"Assistant: <tool_calls>\n{json.dumps(calls, ensure_ascii=False)}\n</tool_calls>")
            else:
                chat_parts.append(f"Assistant: {content}")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            chat_parts.append(f"Tool Response:\n{content}")
            
    prompt_sections = []
    if system_parts:
        prompt_sections.append("System Instructions:\n" + "\n".join(system_parts))
        
    if chat_parts:
        prompt_sections.append("\n\n".join(chat_parts))
        
    return "\n\n".join(prompt_sections)

def parse_model_output(text: str):
    """
    モデルがテキスト出力した <tool_calls> を
    OpenAI 互換の tool_calls 配列構造に変換する
    """
    if not text:
        return None, None
        
    pattern = r"(?:<tool_calls>|<｜｜DSML｜｜ calls>)\s*(.*?)\s*(?:</tool_calls>|</｜｜DSML｜｜ calls>)"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return text, None

    raw_json = match.group(1)
    clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    try:
        calls_data = json.loads(raw_json)
        tool_calls = []

        for idx, call in enumerate(calls_data):
            tool_calls.append({
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": call.get("name"),
                    "arguments": json.dumps(call.get("arguments", {})) if isinstance(call.get("arguments"), dict) else call.get("arguments", "{}")
                }
            })
        return (clean_text if clean_text else None), tool_calls
    except json.JSONDecodeError:
        return text, None

def chat_non_streaming(session_id, messages, model_type="default", thinking_enabled=True, tools=None):
    """Non-streaming chat"""
    user_message = format_messages(messages, tools)
    try:
        chat = get_chat_instance(session_id)
        result = chat.send_message(user_message, printing=False, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
        if result and result.get("ok"):
            raw_resp = result["content"].get("response", "")
            return clean_response(raw_resp)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def chat_streaming(session_id, messages, model, model_type="default", thinking_enabled=True, tools=None):
    """Streaming chat using SSE"""
    user_message = format_messages(messages, tools)
    try:
        chat = get_chat_instance(session_id)
        result = chat.send_message(user_message, printing=False, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
        response_text = ""
        if result and result.get("ok"):
            raw_resp = result["content"].get("response", "")
            response_text = clean_response(raw_resp)
        if not response_text:
            response_text = "Error: Failed to get response from DeepSeek."
    except Exception as e:
        response_text = f"Error: {str(e)}"

    content, tool_calls = parse_model_output(response_text)

    if content:
        chunk_size = 4
        for i in range(0, len(content), chunk_size):
            part = content[i:i+chunk_size]
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": part},
                    "finish_reason": None
                }]
            }
            yield "data: " + json.dumps(chunk) + "\n\n"

    if tool_calls:
        tc_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": tool_calls
                },
                "finish_reason": "tool_calls"
            }]
        }
        yield "data: " + json.dumps(tc_chunk) + "\n\n"
        
    # Final chunk
    stop_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }]
    }
    yield "data: " + json.dumps(stop_chunk) + "\n\n"
    yield "data: [DONE]\n\n"

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    
    messages = data.get("messages", [])
    tools = data.get("tools", None)
    stream = data.get("stream", False)
    # Use 'user' field as session ID, or default to 'default'
    session_id = data.get("user", "default")
    
    # Determine model - default to DeepSeek V3.
    model = data.get("model", "deepseek-v3")
    
    # Determine DeepSeek web model flags based on model name.
    model_type, thinking_enabled = get_model_config(model)
    
    if stream:
        return Response(
            stream_with_context(chat_streaming(session_id, messages, model, model_type, thinking_enabled, tools)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
    else:
        result = chat_non_streaming(session_id, messages, model_type, thinking_enabled, tools)
        content, tool_calls = parse_model_output(result)
        
        message = {"role": "assistant"}
        if content:
            message["content"] = content
        else:
            message["content"] = None
        if tool_calls:
            message["tool_calls"] = tool_calls
            
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(result.split()) if result else 0,
                "total_tokens": len(result.split()) if result else 0
            }
        })

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "deepseek-v3",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek V3 - Fast responses without extended thinking"
            },
            {
                "id": "deepseek-r1",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek R1 - Reasoning model with extended thinking"
            },
            {
                "id": "deepseek-v4",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek V4 - Expert model without extended thinking"
            },
            {
                "id": "deepseek-r4",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek R4 - Expert reasoning model with extended thinking"
            }
        ]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()
    
    print(f"Starting DeepSeek API Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
