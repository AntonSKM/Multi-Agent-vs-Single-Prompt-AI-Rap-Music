# source: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md

import requests
import time
import json
from pathlib import Path
from app.models.llm import audio_prompt_editor_llm

# url for the ace-step api
url = "http://localhost:8001"

def revise_audio_prompt_agent(state):
    # prompt revision agent to revise the given prompt based on critic feedback
    
    print("### audio prompt revision ###")
    
    current_prompt = state["audio_prompt"]
    feedback = state["audio_critique"]

    prompt = f"""
    You are an audio-generation prompt revision editor.

    The audio critic has evaluated the generated song and identified an audible
    artist-persona mismatch.

    [CURRENT AUDIO PROMPT]
    {current_prompt}

    [AUDIO CRITIC FEEDBACK]
    {feedback}

    Revise the current audio prompt only to implement the critic feedback.

    Rules:

    - Treat the critic feedback as the authoritative revision instruction.
    - Preserve all song-specific and already matching characteristics.
    - Correct only the mismatch identified by the critic.
    - Do not independently evaluate the Artist DNA.
    - Do not introduce unrelated stylistic changes.
    - Replace weak or conflicting phrases instead of continuously adding traits.
    - The revised prompt must meaningfully differ from the current prompt.
    - Use concise, positive, directly audible keyword-and-phrase language.
    - Do not include explanations, labels, negative instructions, BPM, duration,
    language, lyrics, or evaluation terminology.
    - Maximum {state["config"]["audio_prompt_characters"]} characters.
    - Return the complete revised audio prompt only.
    """

    # run prompt revision model with the prompt and get cleaned output
    revised_prompt = " ".join(audio_prompt_editor_llm.invoke(prompt).content.strip().split())

    # ace-step only allows 512 character prompts so cleanly cut off at full last keyword if length extends the maximum allowed lenght
    audio_prompt_characters = state["config"]["audio_prompt_characters"]
    if len(revised_prompt) > audio_prompt_characters:
        revised_prompt = (revised_prompt[:audio_prompt_characters].rsplit(" ", 1)[0].rstrip(" ,.;:"))

    return revised_prompt


def audio_agent(state):
    # communicates with ace-step api to generate 
    print("### audio agent ###")

    # get all necessary generation information and settings from state
    audio_decision = state["audio_decision"].strip().upper()
    audio_revision_count = state["audio_revision_count"]
    song_revision_count = state["song_revision_count"]
    lyrics = state["current_lyrics"]
    bpm = int(state["bpm"])
    duration = state["config"].get("duration")
    vocal_language = state["config"].get("vocal_language")
    seed = state["seed"]
    
    # check and conditional for single prompt generation (no audio path shows this is single baseline generation)
    initial_generation = not state["audio_local_path"]
    if initial_generation:
        prompt = state["audio_prompt"]
        
    # conditional branch for MAS generation
    else:
        song_revision_count += 1

        # for regenerate get a revised prompt from prompt revision agent
        if audio_decision == "REGENERATE":
            prompt = revise_audio_prompt_agent(state)
            audio_revision_count += 1

        # if already approved use the same prompt (closest possible output to approved audio as ace-step doesnt allow lyrics change without new generation)
        elif audio_decision == "APPROVED":
            prompt = state["audio_prompt"]
            
        else:
            raise ValueError("wrong output format")

    # ace-step only allows 512 character prompts so cleanly cut off at full last keyword if length extends the maximum allowed lenght
    prompt = " ".join(prompt.split())
    audio_prompt_characters = state["config"]["audio_prompt_characters"]
    if len(prompt) > audio_prompt_characters:
        prompt = prompt[:audio_prompt_characters].rsplit(" ", 1)[0].rstrip(" ,.;:")

    # ace-step settings for the audio generation api call
    ace_step_settings = {
        "model": "acestep-v15-turbo",
        "inference_steps": 50, # recommended amount for high quality generation
        "prompt": prompt,
        "lyrics": lyrics,
        "thinking": True,
        "lm_model_path": "acestep-5Hz-lm-4B",
        "vocal_language": vocal_language,
        "duration": duration,
        "audio_format": "wav",
        "time_signature": "4",  # pretty much standard for rap music
        "batch_size": 1, # generate only 1 song
        "bpm": bpm,
        "task_type": "text2music",
        "use_random_seed": False, # using same seed for single and mas for controlled experiment conditions
        "use_cot_caption": False,
        "infer_method": "ode",
        "seed": seed,
    }

    # post api call for generation
    response = requests.post(f"{url}/release_task", json=ace_step_settings)
    response.raise_for_status()
    task_id = response.json()["data"]["task_id"]
    print(f"song generation task submitted - id {task_id} | {state['condition']} | {state['package_id']} | song version {song_revision_count}")

    # polling for the song generation completion
    while True:
        print("generating song...")

        # query the generation process for status
        query_payload = {"task_id_list": [task_id]}
        query_response = requests.post(f"{url}/query_result", json=query_payload)
        query_response.raise_for_status()
        task_result = query_response.json()["data"][0]
        status = task_result["status"]

        # if task status is 1 audio generation succeeded
        if status == 1:
            
            # get url of generated audio file
            result_details = json.loads(task_result["result"])[0]
            audio_url = f"{url}{result_details['file']}"
            print(f"song successfully generated - url: {audio_url}")

            # save generated audio file in output directory
            folder = Path("generated_audio", state["config"]["experiment_id"], state["condition"], state["package_id"])
            folder.mkdir(parents=True, exist_ok=True)
            file_name = "audio_initial.wav" if song_revision_count == 0 else f"audio_revision_{song_revision_count:02d}.wav"
            local_audio_path = folder / file_name
            audio_response = requests.get(audio_url)
            audio_response.raise_for_status()
            local_audio_path.write_bytes(audio_response.content)

            return {"audio_local_path": str(local_audio_path), "audio_revision_count": audio_revision_count, "song_revision_count": song_revision_count, "audio_prompt": prompt}

        # audio generation failed
        elif status == 2:
            raise RuntimeError("audio generation failed")

        # wait 5 seconds before checking generation status again
        time.sleep(5)
