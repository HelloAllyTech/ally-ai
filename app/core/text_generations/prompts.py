import textwrap
from langchain_core.prompts import PromptTemplate

nudge_prompt = PromptTemplate(
    template=textwrap.dedent("""
        You are an AI assistant that generates nudges for counsellors to respond to their clients.

        Client conversation:
        {conversation}

        Message history:
        {message_history}

        Here is a suggestion, you may only consider the suggestion if and only if it matches with the context:
        {suggestion}

        Output format:
         - Markdown with a header and a brief text explanation under 20 words
    """),
    input_variables=["conversation", "message_history", "suggestion"]
)
