import textwrap
from langchain_core.prompts import PromptTemplate

DIARIZATION_PROMPT = PromptTemplate(
    template=textwrap.dedent("""
        CRITICAL: You must process the ENTIRE transcription provided below. Do not skip, truncate, or omit any part of the conversation.
        
        Please analyze the following conversation transcription between a Client and Counselor and convert it into a structured format with speaker roles, content, and timing information.
                             
        IMPORTANT INSTRUCTIONS:
        1. Process EVERY SINGLE LINE of the transcription - do not skip any content
        2. Identify different speakers in the conversation
        3. Assign consistent role names (Client, Counselor)
        4. Extract the content for each speaker's turn
        5. Use the timing information from the transcription segments to assign start_time and end_time
        6. Translate the text into English language
        7. If the transcription already contains speaker labels (like "Client:", "Counselor:"), use those.                   
        8. If no speaker labels are present, infer speakers based on context and conversation flow.
        9. Group consecutive segments by the same speaker into single messages
        10. Ensure you capture ALL messages from the beginning to the end of the conversation
        11. If the transcription is long, process it systematically from start to finish
        12. Do not stop processing until you have covered the entire transcription
        
        VERIFICATION: Before providing your response, verify that you have processed the entire transcription by checking:
        - You have included messages from the earliest timestamp to the latest timestamp
        - You have not skipped any significant portions of the conversation
        - All timing information is preserved and accurate
        
        Transcription with timing (PROCESS ALL OF THIS):
        ```
        {transcription}
        ```
        
        Remember: Process the ENTIRE transcription from start to finish without any omissions.
    """),
    input_variables=["transcription"]
)
