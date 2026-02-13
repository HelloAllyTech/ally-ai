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
                You're an assistant that creates session notes for mental health
                organizations.
                Analyze the chat history and extract All available information
                for each field in the output schema.
                Avoid leaving fields as null/empty when information
                is present in the conversation.
                For the session summary, provide a detailed summary mostly independent sentences.
                
                      
                Responses shorter than 300 words are invalid.

                IMPORTANT RULES:
                - If the chat history is completely empty or contains no meaningful
                  content, return EXACTLY an empty JSON Object

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


COUNSELOR_ANALYSIS_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        You are a specialized extraction engine for analyzing therapeutic
        counselor communication. Analyze the counselor's message
        and extract
        specific types of therapeutic techniques.

        CRITICAL: Extract EXACT substrings only. Never modify, paraphrase,
        or interpret the text. Copy the counselor's exact words that match
        each category.

        Return ONLY this JSON structure:
        {{
          "reflective": [ "exact quoted text" ],
          "open_ended": [ "exact quoted text" ],
          "back_channel": [ "exact quoted text" ]
        }}

        DETAILED CATEGORY DEFINITIONS:

        1. REFLECTIVE QUESTIONS (Mirroring/Paraphrasing):
           - Questions that reflect back the client's words, feelings, or
             experiences
           - Common patterns:
             * "So you're [feeling/thinking/saying]...?"
             * "It sounds like you're [experiencing/going through]...?"
             * "You mentioned that [client's words]...?"
             * "I hear that [client's experience]...?"
             * "It seems like [client's situation]...?"
           - Examples: "So you're feeling overwhelmed?",
             "It sounds like work is really stressful for you?"

        2. OPEN-ENDED QUESTIONS (Exploratory):
           - Questions that invite detailed, narrative responses
           - Start with: What, How, Why, When, Where, Tell me, Describe,
             Explain, Share, Walk me through, Help me understand
           - Must encourage elaboration and deeper exploration
           - STRICTLY EXCLUDE: Do/Did, Are/Were, Will/Would, Can/Could,
             Should, Have/Has, Is/Was, Does, Have you, Did you
           - Examples: "What does that feel like for you?",
             "How has this been impacting your daily life?",
             "Tell me more about that experience"

        3. BACK-CHANNEL CUES (Active Listening):
           - Brief, supportive responses that show engagement and encourage
             continuation
           - Short acknowledgments that don't ask questions
           - Examples: "I see", "I understand", "That makes sense", "Go on",
             "Mmm-hmm", "I hear you", "That sounds difficult", "I can imagine"

        EXTRACTION RULES:
        - Copy EXACT text only - preserve original wording, punctuation,
          and capitalization
        - If text fits multiple categories, include it in ALL relevant categories
        - If no text matches a category, return empty array []
        - Be conservative - only include clear, unambiguous matches
        - Focus on therapeutic technique, not content analysis

        Counselor Message: {message}
        """
    ),
    input_variables=["message"],
)

SIMULATION_ANALYSIS_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
       You are a clinical supervisor analyzing a counselor training simulation
       where an AI client interacts with a counselor-in-training.

       Evaluate the counselor's performance against the training goal and
       return ONLY a JSON object with two array fields.

       Important rules:
       - Provide specific, actionable feedback points
       - Reference exact examples from the conversation
       - Focus on clinical competencies and therapeutic techniques
       - Each point should be concise but substantive

       Training Goal: {goal}

       Conversation Transcript:
       {chat_history}

       Analyze the counselor's:
       • Therapeutic rapport building and engagement
       • Active listening and reflective techniques
       • Empathy expression and validation
       • Question formulation and timing
       • Professional boundaries and crisis response
       • Goal alignment and skill demonstration

       Return only valid JSON with these fields:
       - "improvements" → Array of specific areas needing development
         with conversation examples
       - "positives" → Array of demonstrated strengths and effective
         techniques with examples

       Return only valid JSON.
       """
    ),
    input_variables=["goal", "chat_history"],
)

SIMULATION_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["chat_history"],
    template=textwrap.dedent(
        """
        You are a clinical supervisor analyzing a transcript of a roleplay between a mental health counselor and a client.
        
        Evaluate the counselor's performance against the following 15 competencies:
        1. Active listening
        2. Verbal communication skills
        3. Explanation and promotion of confidentiality
        4. Rapport building & self-disclosure
        5. Exploration & normalization of feelings
        6. Demonstration of empathy, warmth, & genuineness
        7. Assessment of harm to self, harm to others, harm from others & developing collaborative response plan
        8. Connection to social functioning & impact on life
        9. Exploration of client’s & social support network’s explanation for problem
        10. Appropriate involvement of family members & other close persons
        11. Collaborative goal setting & addressing client’s expectations
        12. Promotion of realistic hope for change
        13. Incorporation of coping mechanisms & prior solutions
        14. Psychoeducation & use of local terminology
        15. Elicitation of feedback when providing advice, suggestions & recommendations

        Conversation Transcript:
        {chat_history}

        Return ONLY a JSON object with exactly two array fields.
        
        Important rules:
        - Provide specific, actionable feedback points based on the 15 competencies above.
        - Reference exact examples or quotes from the conversation to support your points.
        - Each point should be concise but substantive.
        - If a competency was not applicable (e.g., no crisis mentioned), do not force a positive/negative unless the counselor missed an opportunity to address it.
        
        Return only valid JSON with these fields:
        - "positives": Array of demonstrated strengths and effective techniques with examples.
        - "improvements": Array of specific areas needing development with conversation examples.
        
        Return ONLY valid JSON.
        """
    ),
)

SIMULATION_ANALYSIS_WITH_MEMORY_PROMPT = PromptTemplate(
    input_variables=["chat_history", "previous_summary", "custom_prompt_section"],
    template=textwrap.dedent(
        """
        You are a clinical supervisor analyzing a transcript of a roleplay between a mental health counselor and a client.
        You also maintain a comprehensive memory of ongoing client-counsellor interactions.

        {custom_prompt_section}

        Evaluate the counselor's performance against the following 15 competencies:
        1. Active listening
        2. Verbal communication skills
        3. Explanation and promotion of confidentiality
        4. Rapport building & self-disclosure
        5. Exploration & normalization of feelings
        6. Demonstration of empathy, warmth, & genuineness
        7. Assessment of harm to self, harm to others, harm from others & developing collaborative response plan
        8. Connection to social functioning & impact on life
        9. Exploration of client's & social support network's explanation for problem
        10. Appropriate involvement of family members & other close persons
        11. Collaborative goal setting & addressing client's expectations
        12. Promotion of realistic hope for change
        13. Incorporation of coping mechanisms & prior solutions
        14. Psychoeducation & use of local terminology
        15. Elicitation of feedback when providing advice, suggestions & recommendations

        Previous Summary (if available):
        ```
        {previous_summary}
        ```

        Conversation Transcript:
        {chat_history}

        Return ONLY a JSON object with exactly four fields.

        Important rules:
        - Provide specific, actionable feedback points based on the 15 competencies above.
        - Reference exact examples or quotes from the conversation to support your points.
        - Each point should be concise but substantive.
        - If a competency was not applicable (e.g., no crisis mentioned), do not force a positive/negative unless the counselor missed an opportunity to address it.
        - For session_glimpse: Focus ONLY on the current session as a quick snapshot.
        - For cumulative_memory: If a previous summary exists, integrate the new conversation while maintaining historical context and identifying patterns or changes. If no previous summary exists, create a comprehensive initial summary. Track key themes, concerns, therapeutic interventions, progress, setbacks, and emerging patterns. Maintain a coherent narrative showing the evolution of the client's journey.

        Return only valid JSON with these fields:
        - "positives": Array of demonstrated strengths and effective techniques with examples.
        - "improvements": Array of specific areas needing development with conversation examples.
        - "session_glimpse": A brief overview (2-3 sentences) of THIS current session, highlighting main topics, key takeaways, and immediate observations.
        - "cumulative_memory": A comprehensive cumulative narrative (300-500 words) that integrates ALL sessions including the current one, showing progression, patterns, and evolution.

        """
    ),
)

SCENARIO_EVALUATION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "competencies_list"],
    template=textwrap.dedent(
        """
        You are a clinical supervisor analyzing a transcript of a roleplay between a mental health counselor and a client.

        Conversation Transcript:
        {chat_history}

        Competencies to Evaluate:
        {competencies_list}

        Evaluate the counselor's performance and return ONLY a JSON object with exactly six fields.

        Important rules:
        - Provide specific, actionable feedback points based on the competencies above.
        - Reference exact examples or quotes from the conversation to support your points.
        - Each point should be concise but substantive.
        - For achieved_competency_ids: Only include IDs of competencies that were clearly demonstrated in this conversation. Be selective and evidence-based.
        - For message_tags: Tag ONLY the counselor's messages (not the client's messages). For each counselor message, assign applicable tags. Use the exact message ID from the transcript. Only include tags that are clearly relevant to that message.
        - For emotional_movement: Analyze ONLY the client's messages (not the counselor's messages). Rate each client message's emotional state on a scale from -5 (very negative/distressed) to +5 (very positive/happy). Consider the emotional tone, sentiment, and distress level expressed in each message. Use the exact message ID from the transcript.
        - For skill_coverage: Evaluate the counselor's overall skill demonstration across three categories. Assign a percentage (0-100) for each category. Always return exactly three items.
          * "Learning" — Measures the counselor's ability to explore and understand the client's world. Includes: asking open-ended questions, using reflective listening, demonstrating curiosity about the client's experience, exploring underlying feelings, identifying patterns, and gathering relevant information without leading or assuming. Higher scores indicate the counselor actively sought to learn about the client rather than jumping to conclusions.
          * "Support" — Measures the counselor's ability to create a safe, empathetic, and validating environment. Includes: expressing empathy, normalising the client's emotions, affirming the client's strengths, holding emotional space, appropriate pacing, conveying warmth and genuine care, and making the client feel heard and understood. Higher scores indicate the client would feel emotionally supported and not judged.
          * "Standards" — Measures the counselor's adherence to professional and ethical counseling practices. Includes: maintaining appropriate boundaries, avoiding advice-giving or fixing, not projecting personal values, following evidence-based techniques, staying client-centred, using proper session structure, and demonstrating professional language and conduct. Higher scores indicate the counselor maintained high professional standards throughout.

        Return only valid JSON with these fields:
        - "positives": Array of demonstrated strengths and effective techniques with specific examples from the conversation.
        - "improvements": Array of specific areas needing development with conversation examples.
        - "achieved_competency_ids": Array of competency IDs (strings) that were successfully demonstrated. Only include IDs from the provided list above.
        - "message_tags": Array of objects, one per counselor message, each with "id" (the message ID) and "tags" (array of objects with "label" and "category").
        - "emotional_movement": Array of objects, one per client message, each with "message_id" (the message ID) and "level" (integer from -5 to +5).
        - "skill_coverage": Array of exactly 3 objects, each with "category" (one of "Learning", "Support", "Standards") and "percentage" (number from 0 to 100).

        """
    ),
)

SCENARIO_EVALUATION_WITH_MEMORY_PROMPT = PromptTemplate(
    input_variables=["chat_history", "competencies_list", "previous_summary", "custom_prompt_section"],
    template=textwrap.dedent(
        """
        You are a clinical supervisor analyzing a transcript of a roleplay between a mental health counselor and a client.
        You also maintain a comprehensive memory of ongoing client-counselor interactions.

        {custom_prompt_section}

        Conversation Transcript:
        {chat_history}

        Competencies to Evaluate:
        {competencies_list}

        Previous Summary (if available):
        ```
        {previous_summary}
        ```

        Evaluate the counselor's performance and return ONLY a JSON object with exactly eight fields.

        Important rules:
        - Provide specific, actionable feedback points based on the competencies above.
        - Reference exact examples or quotes from the conversation to support your points.
        - Each point should be concise but substantive.
        - For achieved_competency_ids: Only include IDs of competencies that were clearly demonstrated in this conversation. Be selective and evidence-based.
        - For message_tags: Tag ONLY the counselor's messages (not the client's messages). For each counselor message, assign applicable tags. Use the exact message ID from the transcript. Only include tags that are clearly relevant to that message.
        - For emotional_movement: Analyze ONLY the client's messages (not the counselor's messages). Rate each client message's emotional state on a scale from -5 (very negative/distressed) to +5 (very positive/happy). Consider the emotional tone, sentiment, and distress level expressed in each message. Use the exact message ID from the transcript.
        - For skill_coverage: Evaluate the counselor's overall skill demonstration across three categories. Assign a percentage (0-100) for each category. Always return exactly three items.
          * "Learning" — Measures the counselor's ability to explore and understand the client's world. Includes: asking open-ended questions, using reflective listening, demonstrating curiosity about the client's experience, exploring underlying feelings, identifying patterns, and gathering relevant information without leading or assuming. Higher scores indicate the counselor actively sought to learn about the client rather than jumping to conclusions.
          * "Support" — Measures the counselor's ability to create a safe, empathetic, and validating environment. Includes: expressing empathy, normalising the client's emotions, affirming the client's strengths, holding emotional space, appropriate pacing, conveying warmth and genuine care, and making the client feel heard and understood. Higher scores indicate the client would feel emotionally supported and not judged.
          * "Standards" — Measures the counselor's adherence to professional and ethical counseling practices. Includes: maintaining appropriate boundaries, avoiding advice-giving or fixing, not projecting personal values, following evidence-based techniques, staying client-centred, using proper session structure, and demonstrating professional language and conduct. Higher scores indicate the counselor maintained high professional standards throughout.
        - For session_glimpse: Focus ONLY on the current session as a quick snapshot.
        - For cumulative_memory: If a previous summary exists, integrate the new conversation while maintaining historical context and identifying patterns or changes. If no previous summary exists, create a comprehensive initial summary. Track key themes, concerns, therapeutic interventions, progress, setbacks, and emerging patterns. Maintain a coherent narrative showing the evolution of the client's journey.

        Return only valid JSON with these fields:
        - "positives": Array of demonstrated strengths and effective techniques with specific examples from the conversation.
        - "improvements": Array of specific areas needing development with conversation examples.
        - "achieved_competency_ids": Array of competency IDs (strings) that were successfully demonstrated. Only include IDs from the provided list above.
        - "message_tags": Array of objects, one per counselor message, each with "id" (the message ID) and "tags" (array of objects with "label" and "category").
        - "emotional_movement": Array of objects, one per client message, each with "message_id" (the message ID) and "level" (integer from -5 to +5).
        - "skill_coverage": Array of exactly 3 objects, each with "category" (one of "Learning", "Support", "Standards") and "percentage" (number from 0 to 100).
        - "session_glimpse": A brief overview (2-3 sentences) of THIS current session, highlighting main topics, key takeaways, and immediate observations.
        - "cumulative_memory": A comprehensive cumulative narrative (300-500 words) that integrates ALL sessions including the current one, showing progression, patterns, and evolution.

        """
    ),
)
