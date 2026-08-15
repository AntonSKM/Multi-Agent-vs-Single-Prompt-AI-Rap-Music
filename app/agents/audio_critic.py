# source: https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/music_analysis.ipynb

import torch
from app.models.llm import qwen_processor as processor
from app.models.llm import qwen_model as model
from qwen_omni_utils import process_mm_info


def audio_critic_agent(state):
    # checks audio for persona consistency regarding the dna and gives revision feedback
    print("### audio critic ###")

    local_audio_path = state["audio_local_path"]
    # only use the relevant acoustic dna part of the dna for more precision
    artist_dna = state["artist_dna"].split("[ACOUSTIC DNA]", 1)[1].strip()

    prompt = f"""
    You are an audio critic for acoustic artist-persona consistency.

    Judge whether the generated audio plausibly matches the acoustic Artist DNA.
    Evaluate the sound itself, not lyrical meaning, general song quality,
    commercial potential, or personal taste.

    [ACOUSTIC ARTIST DNA]
    {artist_dna}

    BPM is fixed by the experiment and already lies within the Artist-DNA range.
    Never mention, estimate, evaluate, or request changes to BPM or tempo.

    Approve if the dominant vocal and production identity plausibly belongs to
    the defined artist and there is no clear material acoustic persona mismatch.

    Regenerate only if one clearly audible contradiction makes the song sound
    like a different acoustic persona.

    If regeneration is needed, identify exactly one highest-impact audible
    mismatch in only a few concise sentences. Describe what is currently
    audible and what target characteristic should replace or correct it.

    Do not write the audio-generation prompt.

    Output exactly one:

    APPROVED

    or

    REGENERATE: <focused diagnostic feedback>
    """

    # specific input layout for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": local_audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # prepare model input
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    # run qwen3 audio analysis and feedback generation
    with torch.no_grad():
        text_ids = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=128,
            thinker_do_sample=False,
            speaker="Ethan",  # model syntax needs speaker, even when "return_audio = false"
            use_audio_in_video=False,
            return_audio=False,
        )

    # get llm output
    content = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # check critic output and extract decision and critique
    if content.upper() == "APPROVED":
        decision = "APPROVED"
        feedback = ""
    elif content.upper().startswith("REGENERATE:"):
        decision = "REGENERATE"
        feedback = content.split(":", 1)[1].strip()
    else:
        raise ValueError("wrong output format")

    # log critic loop trajectory in history
    audio_history = list(state["audio_history"])
    audio_history.append({"version": state["song_revision_count"], "decision": decision, "critique": feedback, "audio_prompt": state["audio_prompt"], "local_path": state["audio_local_path"]})

    print("decision: " + decision + " | critique: " + feedback)
    return {"audio_decision": decision, "audio_critique": feedback, "audio_history": audio_history}
