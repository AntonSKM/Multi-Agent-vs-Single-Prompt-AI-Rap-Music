from langgraph.graph import StateGraph, END
from app.graph.state import State
from app.agents.lyricist import lyricist_agent
from app.agents.lyrics_critic import lyrics_critic_agent
from app.agents.audio_agent import audio_agent
from app.agents.audio_critic import audio_critic_agent


def check_critic_decision_status(state):
    # check the current status of the critic decisions to decide for conditional edge in graph

    lyrics_approved = state["lyrics_decision"] == "APPROVED"
    audio_approved = state["audio_decision"] == "APPROVED"

    # return decisions based on the status
    if lyrics_approved and audio_approved:
        return "approved"
    if state["song_revision_count"] >= state["config"]["max_song_revisions"]:
        return "limit_reached"
    if not lyrics_approved and not audio_approved:
        return "revise_both"
    if not lyrics_approved:
        return "revise_lyrics"
    return "revise_audio"


# initialize the state graph
workflow = StateGraph(State)

# add agents as nodes
workflow.add_node("lyricist_agent", lyricist_agent)
workflow.add_node("lyrics_critic", lyrics_critic_agent)
workflow.add_node("audio_agent", audio_agent)
workflow.add_node("audio_critic", audio_critic_agent)

# graph structure
workflow.set_entry_point("lyrics_critic")
workflow.add_edge("lyrics_critic", "audio_critic")
# conditional edge for critic loops based on critic decisions
workflow.add_conditional_edges("audio_critic", check_critic_decision_status,
    {
        "approved": END,
        "limit_reached": END,
        "revise_both": "lyricist_agent",
        "revise_lyrics": "lyricist_agent",
        "revise_audio": "audio_agent",
    }
)
workflow.add_edge("lyricist_agent", "audio_agent")
workflow.add_edge("audio_agent", "lyrics_critic")

# compile graph
graph = workflow.compile()
