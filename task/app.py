import os
from dotenv import load_dotenv
load_dotenv()

from task.client import DialClient
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role
from task.prompts import SYSTEM_PROMPT
from task.tools.users.create_user_tool import CreateUserTool
from task.tools.users.delete_user_tool import DeleteUserTool
from task.tools.users.get_user_by_id_tool import GetUserByIdTool
from task.tools.users.search_users_tool import SearchUsersTool
from task.tools.users.update_user_tool import UpdateUserTool
from task.tools.users.user_client import UserClient
from task.tools.web_search import WebSearchTool

DIAL_ENDPOINT = "https://ai-proxy.lab.epam.com"
API_KEY = os.getenv('DIAL_API_KEY')

def main():
    deployment_name = "gpt-4o"
    # deployment_name="gemini-2.5-pro"
    # deployment_name="claude-3-7-sonnet@20250219"

    user_client = UserClient()

    client = DialClient(
        endpoint=DIAL_ENDPOINT,
        deployment_name=deployment_name,
        api_key=API_KEY,
        tools=[
            WebSearchTool(api_key=API_KEY, endpoint=DIAL_ENDPOINT),
            GetUserByIdTool(user_client),
            SearchUsersTool(user_client),
            CreateUserTool(user_client),
            UpdateUserTool(user_client),
            DeleteUserTool(user_client),
        ]
    )

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("Type your question or 'exit' to quit.")
    print("Sample:")
    print("Add Andrej Karpathy as a new user")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        conversation.add_message(Message(Role.USER, user_input))

        ai_message = client.get_completion(conversation.get_messages(), print_request=True)
        conversation.add_message(ai_message)
        print("🤖:", ai_message.content)
        print("=" * 100)
        print()


main()

#TODO:
# Request sample:
# Add Andrej Karpathy as a new user