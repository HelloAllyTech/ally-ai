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

        Important: Always refer to the client as "Client" in your response.
        Do not use pronouns (he/she/they) or any specific names when referring to the client.

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

IDENTIFY_USER_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        You are an AI assistant that identifies whether speaker0 and speaker1 are client or counselor.
        Analyze the following chat history and determine the role of each speaker.

        Chat History:
        ```
        {conversations}
        ```

        Rules:
        1. If the speaker is seeking help, advice, or expressing their feelings/concerns, they are the CLIENT
        2. If the speaker is providing support, asking questions to understand better, or giving professional advice, they are the COUNSELOR
        3. If you cannot determine with confidence, mark as UNKNOWN
        4. Each speaker should be consistently identified throughout the conversation
        5. Look for patterns in the conversation to determine roles

        Important: 
        - Each speaker must be identified as either "client", "counselor", or "unknown"
        - Return only the JSON object, no markdown formatting, no code blocks, no additional text
    """),
    input_variables=["conversations"]
)