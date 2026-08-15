from app.models.llm import lyricist_llm

def lyricist_agent(state):
    # lyricist agent to revise the given lyrics based on critic feedback
    print("### lyricist agent ###")

    prompt = f"""
        You are a professional rap lyricist and revision editor. Your task is to revise the existing lyrics based on the provided critique.

        [CURRENT LYRICS]
        {state["current_lyrics"]}

        [CRITIC FEEDBACK]
        {state["lyrics_critique"]}

        Revise the lyrics only to implement the current critic feedback.
        Treat the critic feedback as the authoritative revision instruction.
        Do not introduce additional stylistic changes based on your own preferences.
        Preserve the topic, narrative perspective, section structure, hook, and all unaffected passages.
        When one line needs revision, rewrite its complete two-to-four-line rhyme or meaning unit when necessary so that the result remains natural and coherent.
        Avoid isolated synonym replacement when it damages rhythm, meaning, rhyme, or conversational phrasing.
        Return the complete revised lyrics only, including the existing section labels.
        Never output something outside the lyrics. Do not include explanations, comments, or any other text.
    """

    # run lyric revision model with prompt
    generated_lyrics = lyricist_llm.invoke(prompt).content.strip()
    revision_count = state["lyrics_revision_count"] + 1

    return {"current_lyrics": generated_lyrics, "lyrics_revision_count": revision_count}
