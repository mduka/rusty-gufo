#!/usr/bin/env python3
"""Renders the Qwen3.6-35B-A3B GGUF chat template
(`reference/qwen36_chat_template.jinja`) the way transformers does
(sandboxed Jinja, trim_blocks, lstrip_blocks, tojson without ASCII escaping)
for the cases the `qwen36_dialect_matches_its_template` test of
`tests/chat_template.rs` renders with `Dialect::Qwen36`, and writes them to
`tests/fixtures/qwen36_chat_template.json`. Needs jinja2.

    python3 src/models/qwen/tests/reference/qwen36_template_goldens.py
"""
import hashlib
import json
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "reference" / "qwen36_chat_template.jinja"
OUTPUT = ROOT / "tests" / "fixtures" / "qwen36_chat_template.json"


def raise_exception(message):
    raise TemplateError(message)


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
RECORD_TOOL = {"type": "function", "function": {
    "name": "record", "description": "Store typed values.",
    "parameters": {"type": "object", "properties": {
        "flag": {"type": "boolean"}, "count": {"type": "integer"},
        "note": {"type": ["string", "null"]}, "tags": {"type": "array"},
        "meta": {"type": "object"}}}}}
HAIKU = [{"role": "user", "content": "Write a haiku about the sea."}]
WEATHER = [{"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "What is the weather in Paris right now?"}]
HISTORY = WEATHER + [
    {"role": "assistant", "content": "", "reasoning_content": "The user asks for weather.",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {
         "name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"}]
TYPED = [{"role": "user", "content": "Record the values."},
         {"role": "assistant", "content": "Recording now.", "tool_calls": [
             {"id": "call_1", "type": "function", "function": {"name": "record", "arguments": {
                 "flag": True, "count": 3, "note": None, "tags": ["a", "b"],
                 "meta": {"k": 1, "nested": [False, None]}}}},
             {"id": "call_2", "type": "function", "function": {"name": "record", "arguments": {
                 "flag": False, "note": "plain text"}}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
         {"role": "tool", "tool_call_id": "call_2", "content": "ok again"},
         {"role": "user", "content": "Thanks. Anything else?"}]
MULTI = [{"role": "user", "content": "Pick a number."},
         {"role": "assistant", "content": "Seven.", "reasoning_content": "The user wants any number."},
         {"role": "user", "content": "Why that one?"}]
INLINE = [{"role": "user", "content": "Pick a number."},
          {"role": "assistant", "content": "<think>\nAny number works.\n</think>\n\nSeven."},
          {"role": "user", "content": "Why that one?"}]
TOOL_LOOP = [{"role": "user", "content": "What is the weather in Paris?"},
             {"role": "assistant", "content": "", "reasoning_content": "Call the tool.",
              "tool_calls": [{"id": "call_1", "type": "function", "function": {
                  "name": "get_weather", "arguments": {"city": "Paris"}}}]},
             {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]
DEVELOPER = [{"role": "system", "content": "Be brief."},
             {"role": "developer", "content": "Answer in English."},
             {"role": "user", "content": "Hi"}]

# name: messages, tools, template kwargs (absent keys stay undefined).
CASES = {
    "plain_off": (HAIKU, None, {"enable_thinking": False}),
    "plain_on": (HAIKU, None, {"enable_thinking": True}),
    "plain_default": (HAIKU, None, {}),
    "effort_low_on": (HAIKU, None, {"enable_thinking": True, "reasoning_effort": "low"}),
    "effort_xhigh_on": (HAIKU, None, {"enable_thinking": True, "reasoning_effort": "xhigh"}),
    "system_tools_off": (WEATHER, [WEATHER_TOOL], {"enable_thinking": False}),
    "system_tools_on": (WEATHER, [WEATHER_TOOL], {"enable_thinking": True}),
    "tool_history_off": (HISTORY, [WEATHER_TOOL], {"enable_thinking": False}),
    "tool_loop_on": (TOOL_LOOP, [WEATHER_TOOL], {"enable_thinking": True}),
    "typed_arguments_on": (TYPED, [RECORD_TOOL], {"enable_thinking": True}),
    "multi_turn_on": (MULTI, None, {"enable_thinking": True}),
    "multi_turn_preserve": (MULTI, None, {"enable_thinking": True, "preserve_thinking": True}),
    "multi_turn_no_preserve": (MULTI, None, {"enable_thinking": True, "preserve_thinking": False}),
    "inline_reasoning_on": (INLINE, None, {"enable_thinking": True}),
    "inline_reasoning_preserve": (INLINE, None, {"enable_thinking": True, "preserve_thinking": True}),
    "system_developer_off": (DEVELOPER, None, {"enable_thinking": False}),
}


def chat(messages):
    """The messages as the Rust test builds `ChatMessage`s: reasoning as the
    thought, tool-call arguments in order as `(name, text, is_string)` with
    non-string values in `tojson` spelling."""
    out = []
    for message in messages:
        calls = []
        for call in message.get("tool_calls", []):
            function = call["function"]
            calls.append({"id": call["id"], "name": function["name"], "arguments": [
                [name, value if isinstance(value, str) else tojson(value), isinstance(value, str)]
                for name, value in function["arguments"].items()]})
        out.append({"role": message["role"], "content": message["content"],
                    "thought": message.get("reasoning_content", ""), "tool_calls": calls})
    return out


def main():
    source = TEMPLATE.read_text()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)
    cases = []
    for name, (messages, tools, kwargs) in CASES.items():
        arguments = {"messages": messages, "add_generation_prompt": True, **kwargs}
        if tools:
            arguments["tools"] = tools
        cases.append({"name": name, "chat": chat(messages),
                      "tools": [json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
                                for tool in tools or []],
                      "kwargs": kwargs, "text": template.render(**arguments)})
    OUTPUT.write_text(json.dumps({
        "template_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source": "tokenizer.chat_template of unsloth Qwen3.6-35B-A3B GGUFs",
        "cases": cases}, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
