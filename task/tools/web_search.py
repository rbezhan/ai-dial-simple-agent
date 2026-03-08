from typing import Any

import requests

from task.tools.base import BaseTool


class WebSearchTool(BaseTool):

    def __init__(self, api_key: str, endpoint: str):
        self.__api_key = api_key
        self.__endpoint = f"{endpoint}/openai/deployments/gemini-2.5-pro/chat/completions"

    @property
    def name(self) -> str:
        return "web_search_tool"

    @property
    def description(self) -> str:
        return "Tool for WEB searching."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for on the web"
                }
            },
            "required": ["request"]
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        headers = {
            "api-key": self.__api_key,
            "Content-Type": "application/json"
        }
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": str(arguments["request"])
                }
            ],
            "tools": [
                {
                    "type": "static_function",
                    "static_function": {
                        "name": "google_search",
                        "description": "Grounding with Google Search",
                        "configuration": {}
                    }
                }
            ],
            "temperature": 0
        }

        response = requests.post(url=self.__endpoint, headers=headers, json=request_data)

        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                return data["error"]
            return data["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} {response.text}"