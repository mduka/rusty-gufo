"""Records the chat template fixture of gufo-deepseek-v4-flash from the C++
engine (`ds4_ref_dump render`): per case the messages, tools and options,
the rendered conversation and the FNV-1a 64 digest and count of its token
IDs.

A case is {"name", "messages": [{"role", "content", "reasoning",
"tool_call_id", "tool_calls": [{"name", "id", "arguments": [{"name",
"value", "string"}]}]}], "tools": [{"name", "description", "parameters",
"definition"}], "options": {"thinking", "effort", "preserve",
"tools_present", "require"}}.

Usage: record_chat_template.py DS4_REF_DUMP MODEL WORKDIR FIXTURE"""
import json, os, subprocess, sys

def fnv(data):
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"

def m(role, content="", **extra):
    message = {"role": role, "content": content}
    message.update(extra)
    return message

def call(name, id, *arguments):
    return {"name": name, "id": id, "arguments": [{"name": n, "value": v, "string": s} for n, v, s in arguments]}

weather = {"name": "get_weather", "description": "Weather for a city, \"now\".\nUse sparingly.",
           "parameters": '{"type":"object","properties":{"city":{"type":"string","description":"Name, e.g. \\"Köln\\""}},"required":["city"]}'}
search = {"name": "search", "description": "Searches the web",
          "definition": '{"name":"search","description":"Searches, ranks: top 10","parameters":{"type":"object","properties":{"q":{"type":"string"},"n":{"type":"integer","minimum":1}}},"strict":true}'}
empty = {"name": "noop", "description": "", "parameters": ""}

cases = [
    {"name": "user", "messages": [m("user", "Hello")]},
    {"name": "system_user", "messages": [m("system", "You are terse."), m("user", "Hi\n\"quoted\" \\ back")]},
    {"name": "thinking_low", "messages": [m("user", "Why?")], "options": {"thinking": True, "effort": "minimal"}},
    {"name": "thinking_high", "messages": [m("system", "S"), m("user", "Why?")], "options": {"thinking": True, "effort": "medium"}},
    {"name": "thinking_max", "messages": [m("user", "Why?")], "options": {"thinking": True, "effort": "xhigh"}},
    {"name": "effort_without_thinking", "messages": [m("user", "Why?")], "options": {"effort": "max"}},
    {"name": "multi_turn_dropped", "messages": [m("user", "One"), m("assistant", "A1", reasoning="R1"), m("user", "Two")],
     "options": {"thinking": True}},
    {"name": "multi_turn_preserved", "messages": [m("user", "One"), m("assistant", "A1", reasoning="R1"), m("user", "Two")],
     "options": {"thinking": True, "preserve": True}},
    {"name": "after_last_user", "messages": [m("user", "One"), m("assistant", "A1", reasoning="R1")],
     "options": {"thinking": True}},
    {"name": "developer_skipped", "messages": [m("developer", "Dev rules"), m("user", "Q"), m("assistant", "A", reasoning="R"),
                                              m("user", "Q2")], "options": {"thinking": True}},
    {"name": "developer_kept", "messages": [m("user", "Q"), m("developer", "Late rules")], "options": {"thinking": True}},
    {"name": "developer_plain", "messages": [m("developer", "Dev"), m("user", "Q")]},
    {"name": "tools_without_host", "messages": [m("user", "Weather in Köln?")], "tools": [weather, search],
     "options": {"tools_present": True}},
    {"name": "tools_required", "messages": [m("user", "Weather?")], "tools": [weather],
     "options": {"tools_present": True, "require": True}},
    {"name": "tools_system_host", "messages": [m("system", "Be helpful."), m("user", "Q")], "tools": [search, empty],
     "options": {"tools_present": True, "thinking": True, "effort": "high"}},
    {"name": "tools_empty_system", "messages": [m("system", ""), m("user", "Q")], "tools": [weather],
     "options": {"tools_present": True}},
    {"name": "tools_developer_host", "messages": [m("developer", "Dev"), m("user", "Q")], "tools": [weather],
     "options": {"tools_present": True}},
    {"name": "tool_calls_sorted", "messages": [
        m("user", "Compare the weather."),
        m("assistant", "", reasoning="Need both.", tool_calls=[
            call("get_weather", "c1", ("city", "Köln", True)),
            call("get_weather", "c2", ("city", "Bonn", True), ("days", "3", False))]),
        m("tool", "Bonn: rain", tool_call_id="c2"),
        m("user", "Also note the time."),
        m("tool", "Köln: sun", tool_call_id="c1"),
        m("tool", "unknown", tool_call_id="zz")], "tools": [weather], "options": {"tools_present": True, "thinking": True}},
    {"name": "tool_call_answer", "messages": [
        m("user", "Weather?"),
        m("assistant", "Checking.", tool_calls=[call("get_weather", "", ("city", "{\"a\": [1, 2]}", False))]),
        m("tool", "sunny"),
        m("assistant", "It is sunny.")], "tools": [weather]},
    {"name": "function_role", "messages": [m("user", "Run it"), m("function", "result: 42")]},
    {"name": "trailing_assistant", "messages": [m("system", "S"), m("user", "Q"), m("assistant", "Partial answer")]},
    {"name": "unicode_controls", "messages": [m("user", "Tabs\tand\rcarriage\u000bvertical, emoji 🦊, CJK 東京")],
     "tools": [{"name": "tést", "description": "ctl\b\f\u0001 chars", "parameters": '{"a":"x:y,z"}'}]},
]

tool, model, work, fixture = sys.argv[1:5]
os.makedirs(work, exist_ok=True)
cases_path = os.path.join(work, "cases.json")
json.dump(cases, open(cases_path, "w", encoding="utf-8"), ensure_ascii=False)
subprocess.run(["choom", "-n", "1000", "--", tool, model, work, "render", cases_path], check=True)
out = {"source": "src/models/deepseek_v4_flash/tests/reference/ds4_ref_dump.cpp render against the C++ engine",
       "cases": []}
for i, case in enumerate(cases):
    rendered = open(os.path.join(work, f"rendered_{i}.txt"), "rb").read().decode("utf-8")
    tokens = open(os.path.join(work, f"tokens_{i}.bin"), "rb").read()
    out["cases"].append({**case, "rendered": rendered, "token_count": len(tokens) // 4, "tokens": fnv(tokens)})
    print(case["name"], len(rendered), "chars", len(tokens) // 4, "tokens", flush=True)
json.dump(out, open(fixture, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
