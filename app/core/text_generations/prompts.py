import textwrap
from langchain_core.prompts import PromptTemplate

NUDGE_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        You are an AI assistant that generates nudges for counsellors to respond to their clients.

        Client conversation:
        ```
        {conversation}
        ```

        Chat history:
        ```
        {chat_history}
        ```
        
        Here is a suggestion from another conversation in this context, 
        you may only consider the suggestion if and only if it matches with the context:
        ```
        {suggestion}
        ```

        Output format:
         - Markdown with a header and a brief text explanation under 20 words
    """),
    input_variables=["conversation", "chat_history", "suggestion"]
)

SUMMARY_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        You're an assistant that creates notes for mental health organisations.
        You'll be provided with the chat history of a conversation between a counselor and client. 
        If chat is empty, return empty values according to the output format.
        Chat history:
        ```
        {chat_history}
        ```
    """),
    input_variables=["chat_history"]
)

CONTENT_ENHANCE_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        You are an AI assistant that enhances notes by mental health counselors
        The content given might be simple points and may contain grammatical errors and may not be well structured.
        Your task is to enhance the content by making it more structured and grammatically correct.
        Use simple language unless it's a technical term.
        If content is empty, return the same.
        
        Output format:
        - Content 1\n- Content 2\n- Content 3
        
        Content:
        ```
        {content}
        ```
    """),
    input_variables=["content"]
)
