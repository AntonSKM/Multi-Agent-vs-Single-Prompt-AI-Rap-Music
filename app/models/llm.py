from langchain_ollama import ChatOllama
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


# ollama models
# source: https://ollama.com/library/qwen3

# creative model for starter pack generation (for weak init run model="qwen3:4b" was used)
starter_pack_llm = ChatOllama(model="qwen3:235b", temperature=0.8, enable_thinking=False)

# lyricist model for revision of the lyrics
lyricist_llm = ChatOllama(model="qwen3:235b", temperature=0.4, enable_thinking=False)

# strict critic model
critic_llm = ChatOllama(model="qwen3:235b", temperature=0.0, enable_thinking=False)

# revision model for ace-step prompt
audio_prompt_editor_llm = ChatOllama(model="qwen3:235b", temperature=0.2, enable_thinking=False)



# qwen3-omni
# sources: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct 
# and https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/music_analysis.ipynb

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
qwen_processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
qwen_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
