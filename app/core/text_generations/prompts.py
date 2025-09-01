import textwrap

from langchain_core.prompts import PromptTemplate

NUDGE_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        You are an AI assistant that generates nudges for counsellors to respond to
        their clients.

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
        Do not use pronouns (he/she/they) or any specific names when referring to
        the client.

        Output format:
         - Markdown with a header and a brief text explanation under 20 words
    """
    ),
    input_variables=["conversation", "chat_history", "suggestion"],
)

SUMMARY_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        You're an assistant that creates session notes for mental health organizations.
        Analyze the chat history and extract All available information
        for each field in the output schema.
        Avoid leaving fields as null/empty when information
        is present in the conversation.
        For the session summary, please provide a summary of the session,
        which has 250 to 500 words.

        Chat history:
        ```
        {chat_history}
        ```
    """
    ),
    input_variables=["chat_history"],
)

CONTENT_ENHANCE_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        You are an AI assistant that enhances notes by mental health counselors
        The content given might be simple points and may contain grammatical
        errors and may not be well structured.
        Your task is to enhance the content by making it more structured and
        grammatically correct.
        Use simple language unless it's a technical term.
        If content is empty, return the same.

        Output format:
        - Content 1\n- Content 2\n- Content 3

        Content:
        ```
        {content}
        ```
    """
    ),
    input_variables=["content"],
)

IDENTIFY_USER_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        You are an AI assistant that identifies whether speaker0 and speaker1
        are client or counselor.
        Analyze the following chat history and determine the role of each speaker.

        Chat History:
        ```
        {conversations}
        ```

        Rules:
        1. If the speaker is seeking help, advice, or expressing their
        feelings/concerns, they are the CLIENT
        2. If the speaker is providing support, asking questions to
        understand better, or giving professional advice, they are
        the COUNSELOR
        3. If you cannot determine with confidence, mark as UNKNOWN
        4. Each speaker should be consistently identified throughout the conversation
        5. Look for patterns in the conversation to determine roles

        Important:
        - Each speaker must be identified as either "client", "counselor", or "unknown"
        - Return only the JSON object, no markdown formatting, no code blocks,
        no additional text
    """
    ),
    input_variables=["conversations"],
)

DYNAMIC_SUMMARY_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        You are an AI assistant that creates structured notes for mental health
        organizations.
        Analyze the following counseling session chat history and provide values
        for ONLY the requested fields.
        Do not include any other fields in your response.

        Requested fields:
        {key_descriptions}

        Chat History:
        ```
        {chat_history}
        ```

        You must return a function call with the following structure:
        {{
            "name": "generate_dynamic_summary",
            "arguments": {{
                "fields": {{
                    "field1": "value1",
                    "field2": 42
                }}
            }}
        }}

        Note:
        - Only include the requested fields
        - For integer fields (like work_life_balance_score), provide integer values
        - For non-numeric fields, provide string values
        - If a field's value cannot be determined, use an empty string
        - Keep values concise and relevant
        - The response must be a valid JSON object
    """
    ),
    input_variables=["chat_history", "key_descriptions"],
)

TAG_POSITIVITY_RATING_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        You are an AI assistant that assigns positivity ratings to
        mental health related tags.

        Analyze the following tags and assign a positivity rating from
        1 to 5 for each tag:
        1 - Highly negative (severe issues, distress)
        2 - Moderately negative (problems, challenges)
        3 - Neutral (neither positive nor negative)
        4 - Moderately positive (improvement, progress)
        5 - Highly positive (success, wellbeing)

        Tags:
        ```
        {tags}
        ```

        Return a list of tags with their positivity ratings.
        Each tag should have the original tag text and a positivity rating.
    """
    ),
    input_variables=["tags"],
)

DIARIZATION_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        CRITICAL: You must process the ENTIRE transcription provided below.
        Do not skip, truncate, or omit any part of the conversation.

        Please analyze the following conversation transcription between a Client
        and Counselor and convert it into a structured format with speaker roles,
        content, and timing information.

        IMPORTANT INSTRUCTIONS:
        1. Process EVERY SINGLE LINE of the transcription - do not skip any content
        2. Identify different speakers in the conversation
        3. Assign consistent role names (Client, Counselor)
        4. Extract the content for each speaker's turn
        5. Use the timing information from the transcription segments to assign
        start_time and end_time.
        6. Translate the text into English language
        7. If the transcription already contains speaker labels
        (like "Client:", "Counselor:"), use those.
        8. If no speaker labels are present, infer speakers based on context
        and conversation flow.
        9. Ensure you capture ALL messages from the beginning to the end of the
        conversation
        10. If the transcription is long, process it systematically from start to finish
        11. Do not stop processing until you have covered the entire transcription
        12. Preserve accurate speaker assignments and timing information
        to enable proper silence calculation

        VERIFICATION: Before providing your response, verify that you have processed
        the entire transcription by checking:
        - You have included messages from the earliest timestamp to the latest
        timestamp
        - You have not skipped any significant portions of the conversation
        - All timing information is preserved and accurate

        Transcription with timing (PROCESS ALL OF THIS):
        ```
        {transcription}
        ```

        Remember: Process the ENTIRE transcription from start to finish without any
        omissions.
    """
    ),
    input_variables=["transcription"],
)
