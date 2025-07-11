import textwrap
from langchain_core.prompts import PromptTemplate

DIARIZATION_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        Please analyze the following conversation transcription between a Client and Counselor and convert it into a structured format with speaker roles, content, and timing information.
                             
        Instructions:
        1. Identify different speakers in the conversation
        2. Assign consistent role names (Client, Counselor)
        3. Extract the content for each speaker's turn
        4. Use the timing information from the transcription segments to assign start_time and end_time
        5. Translate the text into English language
        6. If the transcription already contains speaker labels (like "Client:", "Counselor:"), use those.                   
        7. If no speaker labels are present, infer speakers based on context and conversation flow.
        8. Group consecutive segments by the same speaker into single messages
                             
        Transcription with timing:
        ```
        {transcription}
        ```
    """),
    input_variables=["transcription"]
)
