from typing import Literal, TypedDict


class State(TypedDict, total=False):
    config: dict
    package_id: str
    condition: Literal["single_prompt", "mas"]
    artist_dna: str
    song_topic: str
    current_lyrics: str
    lyrics_critique: str
    lyrics_decision: str
    lyrics_revision_count: int
    lyrics_history: list[dict]
    audio_prompt: str
    bpm: int
    seed: int
    audio_local_path: str
    audio_critique: str
    audio_decision: str
    audio_revision_count: int
    audio_history: list[dict]
    song_revision_count: int
