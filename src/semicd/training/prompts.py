"""Prompt templates frozen to Appendix A.2 of the SemICD paper."""

NOTE_TO_SID = """You are a structured {code_system} coding system.

Given the discharge summary, output the complete set of supported diagnoses using only registered Semantic ID tokens.

Output format:
<SID1_...> <SID2_...> <SID3_...> [UID] <SEP> ...
<SID1_...> <SID2_...> <SID3_...> [UID] <END>

Rules:
- Output Semantic ID tokens only.
- Each diagnosis must contain one SID1 token, one SID2 token, and one SID3 token in that order. Append the registered UID token when the codebook requires it, so that the complete SID identifies exactly one ICD code.
- Separate diagnoses with <SEP>.
- End the sequence with exactly one <END>.
- Do not output ICD codes, diagnosis names, explanations, headings, or any other text.
- Do not output anything after <END>."""

PROFILE_TO_SID = """Convert the following disease profile into its Semantic ID. Output the three registered path tokens, followed by the code-specific UID token if required by the codebook, and then <END>, with no other text."""

SID_TO_PROFILE = """Describe the following Semantic ID using the six fields CHAPTER, BLOCK, CATEGORY, DESCRIPTION, UMLS Terms, and SYNONYMS. The complete Semantic ID identifies exactly one ICD code; output only that code's profile."""

# Raw ICD baseline prompt; only the code system is substituted.
RAW_NOTE = 'You are an {code_system} coding system. Given the discharge summary, output all supported diagnosis codes in canonical order as <code>CODE1,CODE2</code>. Output no explanation.'


def paper_messages(task, text, code_system, representation='semantic', raw_prompt=None, label_space='diagnosis'):
    """Return the paper prompt as system + user chat messages."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompt input must be a nonempty string")
    if representation == 'raw':
        if task != 'note_to_sid' or not raw_prompt:
            raise ValueError('Raw requires the supplied Appendix A.2 prompt and note-only task')
        system = raw_prompt
    elif representation == 'atomic':
        if task == 'note_to_sid':
            system = (f'You are a structured {code_system} coding system. Given the discharge summary, '
                      'output the complete set of supported diagnoses using only registered Atomic ID tokens. '
                      'Each code is one <ATOM_...> token. Separate diagnoses with <SEP> and end with exactly one <END>. '
                      'Do not output ICD codes, explanations or anything after <END>.')
        elif task == 'profile_to_sid':
            system = 'Convert the following disease profile into its registered Atomic ID token followed by <END>, with no other text.'
        elif task == 'sid_to_profile':
            system = 'Describe the following Atomic ID using the six fields CHAPTER, BLOCK, CATEGORY, DESCRIPTION, UMLS Terms, and SYNONYMS. Output only that code\'s profile.'
        else:
            raise ValueError(f'Unsupported prompt task: {task}')
    elif task == "note_to_sid":
        system = NOTE_TO_SID.format(code_system=code_system)
    elif task == "profile_to_sid":
        system = PROFILE_TO_SID
    elif task == "sid_to_profile":
        system = SID_TO_PROFILE
    else:
        raise ValueError(f"Unsupported paper prompt task: {task}")
    if label_space == 'mixed' and representation != 'raw':
        system = system.replace('supported diagnoses', 'supported diagnoses and procedures')
        system = system.replace('Each diagnosis', 'Each code').replace('Separate diagnoses', 'Separate codes')
        system = system.replace('disease profile', 'diagnosis or procedure profile')
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
