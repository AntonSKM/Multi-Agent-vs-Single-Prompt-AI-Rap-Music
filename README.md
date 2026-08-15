# Multi-Agent vs. Single-Prompt AI Rap Music Generation

This repository contains the research prototype developed for the bachelor thesis **“Multi-Agent vs. Single-Prompt AI Rap Music Generation: Evaluating Persona Consistency and Identity Drift Across Multiple Generation Cycles.”**

The project generates complete rap songs from a textual **Artist DNA** and compares two conditions:

1. **Single-Prompt baseline:** one initial song package is rendered once with ACE-Step, without criticism or refinement.
2. **Multi-Agent System (MAS):** the same initial lyrics and byte-identical baseline audio are evaluated by specialized lyrics and audio critics and revised until both critics approve or the shared revision limit is reached.

The repository also contains a post-hoc evaluation script using SentenceTransformer and CLAP embeddings.

> This is a research prototype created for controlled experiments. It is not intended as a production-ready music application.

## System overview

The generation workflow uses:

- **Qwen3-235B-A22B through Ollama** for initial package generation, lyrics criticism, lyric revision, and audio-prompt revision
- **Qwen3-Omni-30B-A3B-Instruct** for direct audio criticism
- **ACE-Step 1.5** for complete song generation
- **LangGraph** for the critic and revision workflow
- **SentenceTransformer `all-mpnet-base-v2`** for lyrical Artist-DNA similarity
- **CLAP `laion/clap-htsat-unfused`** for acoustic Artist-DNA similarity

The main workflow is:

```text
create initial song package
        |
        v
render Single-Prompt baseline with ACE-Step
        |
        v
copy baseline audio as MAS version 0
        |
        v
lyrics critic -> audio critic -> routing decision
        |
        +-- both approved ----------------------> stop
        +-- revision limit reached -------------> stop
        +-- lyrics revision requested ----------> lyricist -> audio agent
        +-- audio regeneration requested -------> audio agent
        +-- both requested ----------------------> lyricist -> audio agent
                                                     |
                                                     v
                                             return to both critics
```

## Repository structure

```text
.
├── app/
│   ├── agents/
│   │   ├── audio_agent.py
│   │   ├── audio_critic.py
│   │   ├── lyricist.py
│   │   └── lyrics_critic.py
│   ├── graph/
│   │   ├── state.py
│   │   └── workflow.py
│   ├── models/
│   │   └── llm.py
│   ├── experiment_setup.py
│   └── main.py
├── config.json
├── evaluate_experiment.py
├── requirements.txt
└── generated_audio/
```

## Important environment note

The main project and ACE-Step should use **separate Python virtual environments**.

ACE-Step uses a different dependency stack, especially a different Transformers version. Installing the ACE-Step dependencies into the main project environment can break Qwen3-Omni, while installing the main project dependencies into the ACE-Step environment can break ACE-Step.

Recommended setup:

```text
main project venv
├── LangGraph
├── Ollama client
├── Qwen3-Omni
├── SentenceTransformer
└── CLAP evaluation

ACE-Step venv
└── ACE-Step API and its own dependencies
```

Do not merge both environments unless the dependency versions have been tested together.

## Requirements

The setup used for the thesis required:

- Linux server
- Python 3.10 or newer
- NVIDIA GPUs with CUDA
- a local Ollama installation
- a separate ACE-Step 1.5 checkout
- FFmpeg libraries required by ACE-Step
- enough GPU memory for Qwen3-235B, Qwen3-Omni, and ACE-Step

The thesis experiment was run on eight NVIDIA RTX A6000 GPUs. GPUs 0–6 were used for the language and multimodal models, while GPU 7 was used for ACE-Step. This is the tested setup, not a formally established minimum hardware requirement.

## Main project environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The provided `requirements.txt` represents the environment used during development. It may contain more packages than the core project directly imports.

Make sure the required Ollama model is available:

```bash
ollama pull qwen3:235b
```

Qwen3-Omni is loaded through Hugging Face when the main application starts. The first start may therefore require substantial download time and storage.

## ACE-Step environment

Create and use a separate environment inside the ACE-Step repository:

```bash
cd /path/to/ACE-Step-1.5
python -m venv .venv_acestep
source .venv_acestep/bin/activate
pip install --upgrade pip
```

Install the ACE-Step requirements according to its repository instructions. On the thesis server, the requirements were installed **without `flash-attn`** because it was not required for this setup and caused installation problems.

The exact ACE-Step dependency versions should remain isolated from the main project environment.

## Configuration

All main experiment settings are defined in `config.json`.

Example:

```json
{
  "artist_dna": "[LYRICAL DNA] ... [ACOUSTIC DNA] ...",
  "song_count": 30,
  "experiment_id": "exp_110",
  "max_song_revisions": 5,
  "duration": 150,
  "vocal_language": "en",
  "seed": 1000,
  "audio_prompt_characters": 512
}
```

### Important configuration rules

- Change `experiment_id` before every new run.
- Reusing an existing ID can overwrite files in the existing experiment directory.
- The package seed is calculated as `seed + song index`.
- `audio_prompt_characters` should remain at or below 512 for the tested ACE-Step API.
- Longer audio prompts are automatically shortened at the last complete word boundary.
- The same configuration is used for both experimental conditions.

## Artist DNA format

The current code requires these exact section markers:

```text
[LYRICAL DNA]
...

[ACOUSTIC DNA]
...
```

Do not rename or remove them. The critics and evaluator split the Artist DNA using these exact strings.

A separate `[GENERAL ARTIST IDENTITY]` section is not used by the critics or the evaluator. It may affect initial package generation if placed before `[LYRICAL DNA]`, but it would not be part of the independent lyrical or acoustic evaluation target. For controlled runs, the complete relevant persona specification should therefore be placed inside the two required sections.

### Lyrical DNA guidance

The Lyrical DNA should describe multiple parameters. Specifications could include for example:

- **Narrative perspective:** first person, third person, autobiographical, detached, observational, or fictional
- **Themes:** recurring subjects, conflicts, situations, and experiences
- **Vocabulary:** direct, poetic, conversational, regional, explicit, metaphorical, simple, or complex language
- **Emotional attitude:** confidence, aggression, restraint, vulnerability, distrust, humor, or reflection
- **Rhyme behavior:** rhyme structures, internal rhymes, multisyllabic patterns, and rhyme density
- **Hook style:** repetition, length, melodic behavior, directness, and concept focus
- **Flow characteristics:** cadence, rhythm, line length, articulation, melodic movement, and vocal pocket

### Acoustic DNA guidance

The Acoustic DNA should describe multiple parameters. Specifications could include for example:

- **Voice register:** low, mid-range, high, or variable
- **Vocal timbre:** dark, bright, heavy, thin, rough, smooth, raspy, or clean
- **Vocal delivery:** aggressive, restrained, melodic, conversational, energetic, or laid-back
- **Vocal processing:** autotune, saturation, compression, reverb, layering, stereo position, and dryness
- **Instrumentation:** instruments, samples, synths, textures, and recurring sound elements
- **Drums and bass:** kicks, snares, hi-hats, percussion, bass, 808s, groove, and rhythmic behavior
- **Arrangement:** intros, hooks, verses, transitions, repetition, negative space, and energy development
- **Tempo:** BPM range or general tempo feeling
- **Mixing and mastering:** vocal placement, stereo width, low end, transients, saturation, loudness, and dynamics
- **Atmosphere:** moods, environments, visual associations, and overall sonic aesthetic

Keep the Artist DNA specific enough to define a recognizable fictional persona, but avoid real artist names and direct imitation instructions. Different characteristics than the ones named here can be used as well and there is no limited set of pre-defined possibilities.

## Starting the system

The tested server setup used three terminals.

### Terminal 1: start Ollama

```bash
cd /path/to/ollama
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bin/ollama serve
```

Without manual GPU assignment:

```bash
bin/ollama serve
```

Keep this terminal running.

### Terminal 2: start the ACE-Step API

```bash
cd /path/to/ACE-Step-1.5
source .venv_acestep/bin/activate
export LD_LIBRARY_PATH=/path/to/ACE-Step-1.5/ffmpeg_shared/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=7 ./start_api_server.sh
```

Without manual GPU assignment:

```bash
./start_api_server.sh
```

The project expects the ACE-Step API at:

```text
http://localhost:8001
```

Keep this terminal running and wait until the API is ready before starting the experiment.

### Terminal 3: run the experiment

From the project root:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m app.main
```

Without manual GPU assignment:

```bash
python -m app.main
```

The application will:

1. load `config.json`
2. generate all initial song packages
3. save `config.json` and `initial_packages.json` inside the experiment directory
4. generate the Single-Prompt baseline for each package
5. copy the baseline audio as MAS version 0
6. run the LangGraph critic and revision workflow
7. save the final outputs and complete critic histories

## Output structure

Experiment artifacts are stored under:

```text
generated_audio/<experiment_id>/
```

Example:

```text
generated_audio/exp_110/
├── config.json
├── initial_packages.json
├── single_prompt/
│   ├── song_01/
│   │   ├── audio_initial.wav
│   │   └── result.json
│   └── ...
└── mas/
    ├── song_01/
    │   ├── audio_initial.wav
    │   ├── audio_revision_01.wav
    │   ├── audio_revision_02.wav
    │   └── result.json
    └── ...
```

The MAS `audio_initial.wav` is a byte-identical copy of the paired Single-Prompt audio. Every later audio revision is stored as a separate WAV file.

Each `result.json` contains:

- experiment and package metadata
- final lyrics, audio prompt, BPM, and audio filename
- shared song revision count
- final critic decisions
- modality-specific revision counts
- complete lyrics and audio critic histories
- critic feedback and version references

## Evaluation

Activate the main project environment and run:

```bash
source .venv/bin/activate
python3 evaluate_experiment.py generated_audio/<experiment_id>
```

Example:

```bash
python3 evaluate_experiment.py generated_audio/exp_110
```

The evaluator first validates the paired experiment structure and then computes the thesis metrics.

### Evaluation settings

The fixed evaluation setup uses:

- `sentence-transformers/all-mpnet-base-v2`
- `laion/clap-htsat-unfused`
- normalized text embeddings
- mono audio loaded at 48 kHz
- 10-second audio segments
- L2 normalization of each segment embedding
- element-wise mean pooling across segments
- final L2 normalization of the song embedding
- paired MAS-minus-Single-Prompt differences
- 5,000 bootstrap samples with seed 2026

### Evaluation outputs

The evaluator writes its results to:

```text
generated_audio/<experiment_id>/evaluation/
```

Outputs include:

```text
evaluation/
├── evaluation_summary.json
├── metrics_per_song.csv
├── paired_summary.csv
├── lyrics_loop.csv
├── audio_loop.csv
├── revision_summary.csv
├── intragroup_similarity.csv
└── plots/
    ├── lyrics_paired_differences.png
    ├── audio_paired_differences.png
    ├── lyrics_revision_trajectories.png
    └── audio_revision_trajectories.png
```

The computational evaluator does not analyze the separate human listening study.

## Main reported metrics

Lyrics and audio are evaluated separately.

The evaluator calculates:

- final Artist-DNA cosine similarity
- paired MAS-minus-Single-Prompt differences
- mean and median paired differences
- 95% bootstrap confidence intervals
- package-level MAS, Single-Prompt, and tie counts
- initial-to-final similarity gain
- positive and negative revision-step fractions
- final-is-best fraction
- best-minus-final difference
- final critic approval rates
- average modality-specific revision counts
- shared revision-limit rate
- supplementary intra-group similarity

The embedding scores are proxy measures of Artist-DNA alignment. They are not percentages and should not be interpreted as complete measures of artistic identity, musical quality, or listener preference.

## Reproducing the thesis setup

For a run that should remain comparable to the thesis experiment:

- keep the model names and temperatures unchanged
- keep Qwen3 thinking disabled
- keep the ACE-Step settings unchanged
- use the same Artist DNA for both conditions
- do not edit generated initial packages after creation
- keep the same package-specific seed in both conditions
- use the copied Single-Prompt audio as MAS version 0
- keep the same revision limit and evaluation settings
- use a new `experiment_id`

The system supports procedural reproducibility and complete artifact preservation. Bitwise-identical regeneration of newly generated model outputs is not guaranteed across different software, hardware, or runtime environments.

## Troubleshooting

### Qwen3-Omni or Transformers import errors

Confirm that the main project environment is active and that ACE-Step dependencies were not installed into it:

```bash
source .venv/bin/activate
```

### ACE-Step dependency conflicts

Use the separate ACE-Step environment:

```bash
cd /path/to/ACE-Step-1.5
source .venv_acestep/bin/activate
```

### ACE-Step connection error

Confirm that the API is running on port 8001:

```text
http://localhost:8001
```

### Ollama connection error

Confirm that `ollama serve` is running and that `qwen3:235b` is installed.

### FFmpeg or shared-library error

Set the ACE-Step library path before starting the API:

```bash
export LD_LIBRARY_PATH=/path/to/ACE-Step-1.5/ffmpeg_shared/lib:$LD_LIBRARY_PATH
```

### CUDA out-of-memory error

Reduce the number of simultaneously visible models or adjust GPU allocation. The tested thesis setup assigned GPUs 0–6 to Ollama and the main application and GPU 7 to ACE-Step.

### Existing experiment files are overwritten

Stop the run and assign a new value to `experiment_id` in `config.json`.

## Thesis

This code accompanies the bachelor thesis:

> Anton Kunstmann. _Multi-Agent vs. Single-Prompt AI Rap Music Generation: Evaluating Persona Consistency and Identity Drift Across Multiple Generation Cycles._ Goethe University Frankfurt, 2026.
