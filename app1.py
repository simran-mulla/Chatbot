import streamlit as st
import validators
import os
import pickle
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv

# ✅ Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Streamlit UI
st.set_page_config(page_title="LangChain: Summarize YT or Website", page_icon="🦜")
st.title("🦜 Summarize YouTube Video or Website")
st.subheader("Enter a YouTube or Website URL:")

generic_url = st.text_input("Enter URL here:")

# ✅ Pickled Model Configuration (Storing Only Model Name, Not Full Object)
MODEL_CONFIG_FILE = "llm_config.pkl"

if os.path.exists(MODEL_CONFIG_FILE):
    with open(MODEL_CONFIG_FILE, "rb") as file:
        try:
            model_config = pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            model_config = {"model": "llama3-8b-8192"}
else:
    model_config = {"model": "llama3-8b-8192"}
    with open(MODEL_CONFIG_FILE, "wb") as file:
        pickle.dump(model_config, file)

# ✅ Load LLM Model
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model=model_config["model"], groq_api_key=GROQ_API_KEY)

# ✅ Function to Get YouTube Transcript
def get_youtube_transcript(video_url):
    """Extracts transcript from a YouTube video."""
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])

        transcript_text = " ".join([entry["text"] for entry in transcript_data])
        return transcript_text

    except TranscriptsDisabled:
        return "⚠️ Transcripts are disabled for this video."
    except Exception as e:
        return f"⚠️ Error fetching transcript: {str(e)}"

# ✅ Prompt Template
prompt_template = """
Provide a concise summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ✅ Summarization Logic
if st.button("Summarize the Content"):
    if not generic_url.strip():
        st.error("🚨 Please enter a valid URL to proceed.")
    elif not validators.url(generic_url):
        st.error("❌ Invalid URL! Please enter a correct YouTube or website URL.")
    else:
        try:
            with st.spinner("⏳ Fetching and summarizing content..."):
                docs = None  # Default

                # ✅ Process YouTube Video
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    transcript = get_youtube_transcript(generic_url)
                    if "⚠️" in transcript:
                        st.error(transcript)
                        docs = None
                    else:
                        docs = [transcript]

                # ✅ Process Website
                else:
                    web_loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        }
                    )
                    docs = web_loader.load()

                # ✅ Handle Empty Content
                if not docs:
                    st.error("⚠️ Failed to retrieve content. The page might be blocking requests.")
                else:
                    # ✅ Summarization Chain
                    chain = load_summarize_chain(st.session_state.llm, chain_type="stuff", prompt=prompt)
                    response = chain.invoke(docs)

                    summary = response.get("output_text", str(response))

                    # ✅ Display Summary
                    st.success("✅ Summary Generated Successfully!")
                    st.write(summary)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
