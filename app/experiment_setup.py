from app.graph.workflow import graph
from app.models.llm import starter_pack_llm
import json
from pathlib import Path
from app.agents.audio_agent import audio_agent
import shutil


def save_result(result, config, system_name, package):
    # save the results with all relevant data of the run in result.json
    
    print(f"saving result for {system_name} - {package['package_id']}")
    folder = Path("generated_audio", config["experiment_id"], system_name, package["package_id"])
    folder.mkdir(parents=True, exist_ok=True)
    audio_file_name = Path(result["audio_local_path"]).name
    critics_used = system_name == "mas"

    # final json structure with experiment data
    final_result = {
        "experiment_id": config["experiment_id"],
        "condition": system_name,
        "package_id": package["package_id"],
        "topic": package["topic"],
        "seed": package["seed"],
        "song_revision_count": result.get("song_revision_count", 0),
        "final_output": {
            "lyrics": result.get("current_lyrics", package["lyrics"]),
            "audio_prompt": result.get("audio_prompt", package["audio_prompt"]),
            "bpm": result.get("bpm", package["bpm"]),
            "audio_file_name": audio_file_name,
        },
        "lyrics_critic": {
            "decision": (result.get("lyrics_decision") if critics_used else "NOT_APPLICABLE"),
            "revision_count": result.get("lyrics_revision_count", 0),
            "critique": result.get("lyrics_critique", ""),
            "history": result.get("lyrics_history", []),
        },
        "audio_critic": {
            "decision": (result.get("audio_decision") if critics_used else "NOT_APPLICABLE"),
            "revision_count": result.get("audio_revision_count", 0),
            "critique": result.get("audio_critique", ""),
            "history": result.get("audio_history", []),
        },
    }

    result_path = folder / "result.json"

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(final_result, file, ensure_ascii=False, indent=2)

def generate_package(config, song_index, previous_topics):
    print(f"generating initial song package {song_index + 1}")
    
    artist_dna = config["artist_dna"]

    # get previous topics to avoid re use of the same topic
    if previous_topics:
        previous_topics_text = "\n".join(f"- {topic}" for topic in previous_topics)
    else:
        previous_topics_text = "No previous topics exist. This is the first package."

    prompt = f"""
        You are a rap songwriter and music producer.

        Your task is to create one song package for the given artist.

        [ARTIST DNA]
        "{artist_dna}"

        First, choose one new song topic that fits the artist and differs fundamentally from the previous topics.
        
        [PREVIOUS TOPICS]
        "{previous_topics_text}"

        Then create:

        1. Complete rap lyrics.
        2. One audio prompt.
        3. One fitting integer BPM value.

        Use exactly this structure:

        [Verse 1]
        [Chorus]
        [Verse 2]
        [Chorus]
        [Outro]

        AUDIO PROMPT RULES:
        - Make the audio prompt directly usable by the audio model, maximum {config["audio_prompt_characters"]} characters.
        - Use a short keyword-and-phrase format to describe the song.
        - Do not include lyrics or BPM.
        - NEVER use more than {config["audio_prompt_characters"]} characters for the audio prompt!

        Return valid JSON only in exactly this structure:

        {{
            "topic": "specific song topic",
            "lyrics": "complete lyrics",
            "audio_prompt": "sonic prompt",
            "bpm": integer BPM value
        }}
        
        Never output anything outside the JSON object.
        """


    # run llm with the prompt
    content = starter_pack_llm.invoke(prompt).content.replace("```json", "").replace("```", "").strip()
    generated = json.loads(content)
    generated["bpm"] = int(generated["bpm"])

    # ace-step only allows 512 character prompts so cleanly cut off at full last keyword if length extends the maximum allowed lenght
    generated["audio_prompt"] = " ".join(generated["audio_prompt"].split())
    audio_prompt_characters = config["audio_prompt_characters"]
    if len(generated["audio_prompt"]) > audio_prompt_characters:
        generated["audio_prompt"] = (generated["audio_prompt"][:audio_prompt_characters].rsplit(" ", 1)[0].rstrip(" ,.;:"))


    return {"package_id": f"song_{song_index + 1:02d}", "topic": generated["topic"], "lyrics": generated["lyrics"], "audio_prompt": generated["audio_prompt"], "bpm": generated["bpm"],"seed": int(config["seed"]) + song_index}


def generate_initial_packages(config):
    packages = []
    previous_topics = []

    # generate song_count# initial packages for the run
    for index in range(config["song_count"]):
        package = generate_package(config, index, previous_topics)
        packages.append(package)
        previous_topics.append(package["topic"])

    # save config and initial packages in seperate experiment folder
    output_folder = Path("generated_audio", config["experiment_id"])
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    with open(output_folder / "initial_packages.json", "w", encoding="utf-8") as file:
        json.dump(packages, file, ensure_ascii=False, indent=2)

    return packages

def create_state(config, package, condition):
    
    # helper function to create a starting state for each system run
    return {
        "config": config,
        "package_id": package["package_id"],
        "condition": condition,
        "artist_dna": config["artist_dna"],
        "song_topic": package["topic"],
        "current_lyrics": package["lyrics"],
        "lyrics_critique": "",
        "lyrics_decision": "INITIAL",
        "lyrics_revision_count": 0,
        "lyrics_history": [],
        "audio_prompt": package["audio_prompt"],
        "bpm": package["bpm"],
        "seed": package["seed"],
        "audio_local_path": "",
        "audio_critique": "",
        "audio_decision": "INITIAL",
        "audio_revision_count": 0,
        "audio_history": [],
        "song_revision_count": 0
    }


def run_experiment(config, packages):
    # run the experiment by first generating single prompt audio and then running MAS for each initial song package
    
    for index, package in enumerate(packages):
        print(f"song package {index + 1:02d}")
        print("### running single prompt generation ###")

        # run the single prompt audio generation for the package and save result
        single_state = create_state(config, package, "single_prompt")
        single_state.update(audio_agent(single_state))
        save_result(single_state, config, "single_prompt", package)

        # copy the exact baseline audio into the MAS folder
        mas_state = create_state(config, package, "mas")
        mas_folder = Path("generated_audio",config["experiment_id"], "mas", package["package_id"])
        mas_folder.mkdir(parents=True, exist_ok=True)
        mas_initial_audio = mas_folder / "audio_initial.wav"
        shutil.copy2(single_state["audio_local_path"], mas_initial_audio)
        mas_state["audio_local_path"] = str(mas_initial_audio)

        # run the MAS langgraph workflow with critics for the package and save result
        print("### running MAS ###")
        result = graph.invoke(mas_state)
        save_result(result, config, "mas", package)

    print("\n### experiment finished ###")

