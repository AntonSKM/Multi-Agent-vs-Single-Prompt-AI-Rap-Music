from app.models.llm import critic_llm


def lyrics_critic_agent(state):
    # checks lyrics for persona consistency regarding the dna and gives revision feedback
    print("### lyrics critic ###")

    revision_count = state["song_revision_count"]
    history = list(state["lyrics_history"])
    
    # check to never revise already approved lyrics
    if history:
        last_entry = history[-1]
        same_lyrics = last_entry.get("lyrics", "").strip() == state["current_lyrics"].strip()
        already_approved = last_entry.get("decision") == "APPROVED"

        if already_approved and same_lyrics:
            if last_entry.get("version") != revision_count:
                history.append({"version": revision_count, "decision": "APPROVED", "critique": "", "lyrics": state["current_lyrics"]})

            print("decision: APPROVED | critique: ")
            return {"lyrics_decision": "APPROVED", "lyrics_critique": "", "lyrics_history": history}

    # only use the relevant lyrical dna part of the dna for more precision
    artist_dna = (state["artist_dna"].split("[LYRICAL DNA]", 1)[1].split("[ACOUSTIC DNA]", 1)[0].strip())

    prompt = f"""
    You are a rap lyrics critic for artist-persona consistency.

    Judge whether the current lyrics plausibly belong to the lyrical Artist DNA.
    Do not judge general writing quality, grammar, personal taste, or minor
    wording preferences.

    [LYRICAL ARTIST DNA]
    {artist_dna}

    [SONG TOPIC]
    {state["song_topic"]}

    [CURRENT LYRICS]
    {state["current_lyrics"]}

    Approve if the complete lyrics have no clear material persona inconsistency.

    Request revision only for one dominant issue that clearly affects persona,
    narrative perspective, emotional attitude, register, topic alignment, hook,
    or flow identity.

    Do not revise isolated synonyms, rhyme words, syllable counts, grammar,
    minor slang preferences, or a passage that is already reasonably corrected.

    Output exactly one:

    APPROVED

    or

    REVISE: <one concise instruction for the lyric revisionist>
    """

    # run critic with the prompt
    content = critic_llm.invoke(prompt).content.strip()

    # check critic output and extract decision and critique
    if content.upper() == "APPROVED":
        decision = "APPROVED"
        critique = ""
    elif content.upper().startswith("REVISE:"):
        decision = "REVISE"
        critique = content.split(":", 1)[1].strip()
    else:
        raise ValueError("wrong output format")

    # log critic loop trajectory in history
    history.append({"version": state["song_revision_count"], "decision": decision, "critique": critique, "lyrics": state["current_lyrics"]})
    
    print("decision: " + decision + " | critique: " + critique)
    return {"lyrics_decision": decision, "lyrics_critique": critique, "lyrics_history": history}
