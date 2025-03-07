import textwrap
from langchain_core.prompts import PromptTemplate

NUDGE_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        You are an AI assistant that generates nudges for counsellors to respond to their clients.

        Client conversation:
        {conversation}

        Chat history:
        {chat_history}

        Here is a suggestion, you may only consider the suggestion if and only if it matches with the context:
        {suggestion}

        Output format:
         - Markdown with a header and a brief text explanation under 20 words
    """),
    input_variables=["conversation", "chat_history", "suggestion"]
)

SUMMARY_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        You're an assistant that creates notes for mental health organisations.
        You'll be provided with the chat history of a conversation between a counselor and client. 
        Chat history:
        {chat_history}
    """),
    input_variables=["chat_history"]
)
