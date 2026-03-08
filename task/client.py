import json
from typing import Any

import requests

from task.models.message import Message
from task.models.role import Role
from task.tools.base import BaseTool


class DialClient:

    def __init__(
            self,
            endpoint: str,
            deployment_name: str,
            api_key: str,
            tools: list[BaseTool] | None = None
    ):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self.__endpoint = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions"
        self.__api_key = api_key
        self.__tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools} or {}
        self.__tools_schemas: list[str]= [tool.schema for tool in tools] or []

        print(self.__endpoint)
        print(json.dumps(self.__tools_schemas, indent=4))


    def get_completion(self, messages: list[Message], print_request: bool = True) -> Message:
        headers = {
            "api-key": self.__api_key,
            "Content-Type": "application/json"
        }
        request_data = {
            "messages": [msg.to_dict() for msg in messages],
            "tools": self.__tools_schemas,
            "stream": True,
        }

        if print_request:
            print(self.__endpoint)
            print("REQUEST:", json.dumps({"messages": [msg.to_dict() for msg in messages]}, indent=2))

        response = requests.post(url=self.__endpoint, headers=headers, json=request_data, stream=True)

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        content_parts: list[str] = []
        # tool_calls_map: index -> {"id": str, "name": str, "arguments": str}
        tool_calls_map: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None

        print("\n\n\n\n🤖: ", end="", flush=True)
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            if not text.startswith("data: "):
                continue
            payload = text[len("data: "):]
            if payload == "[DONE]":
                break

            chunk = json.loads(payload)
            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason") or finish_reason

            # Accumulate content
            if delta.get("content"):
                token = delta["content"]
                content_parts.append(token)
                print(token, end="", flush=True)

            # Accumulate tool call deltas
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta["index"]
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                entry = tool_calls_map[idx]
                if tc_delta.get("id"):
                    entry["id"] += tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    entry["name"] += fn["name"]
                if fn.get("arguments"):
                    entry["arguments"] += fn["arguments"]

        print()  # newline after streamed content

        content = "".join(content_parts) or None
        tool_calls: list[dict[str, Any]] | None = None
        if tool_calls_map:
            tool_calls = [
                {
                    "id": entry["id"],
                    "type": "function",
                    "function": {
                        "name": entry["name"],
                        "arguments": entry["arguments"],
                    },
                }
                for entry in tool_calls_map.values()
            ]
            print("TOOL CALLS:", json.dumps(tool_calls, indent=2))
            print("-" * 100)

        ai_response = Message(role=Role.AI, content=content, tool_calls=tool_calls)

        if finish_reason == "tool_calls":
            messages.append(ai_response)
            tool_messages = self._process_tool_calls(tool_calls)
            messages.extend(tool_messages)
            return self.get_completion(messages, print_request)

        return ai_response


    def _process_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[Message]:
        """Process tool calls and add results to messages."""
        tool_messages = []
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function = tool_call["function"]
            function_name = function["name"]
            arguments = json.loads(function["arguments"])

            tool_execution_result = self._call_tool(function_name, arguments)

            tool_messages.append(Message(
                role=Role.TOOL,
                name=function_name,
                tool_call_id=tool_call_id,
                content=tool_execution_result
            ))

            print(f"FUNCTION '{function_name}'\n{tool_execution_result}\n{'-'*50}")

        return tool_messages

    def _call_tool(self, function_name: str, arguments: dict[str, Any]) -> str:
        tool = self.__tools_dict.get(function_name)
        if tool:
            return tool.execute(arguments)

        return f"Unknown function: {function_name}"
