import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv
import os
import pickle

# ✅ Load API Key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Streamlit UI
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize any YouTube Video or Website")

# ✅ File path for pickled model config (not full LLM)
MODEL_CONFIG_FILE = "llm_config.pkl"

# ✅ Load or Initialize LLM (Only store config, not full object)
if os.path.exists(MODEL_CONFIG_FILE):
    try:
        with open(MODEL_CONFIG_FILE, "rb") as file:
            model_config = pickle.load(file)
    except (EOFError, pickle.UnpicklingError):
        model_config = {"model": "llama3-8b-8192"}
else:
    model_config = {"model": "llama3-8b-8192"}
    with open(MODEL_CONFIG_FILE, "wb") as file:
        pickle.dump(model_config, file)

# ✅ Reinitialize LLM (since full object can't be pickled)
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model=model_config["model"], groq_api_key=GROQ_API_KEY)

# ✅ User Input
generic_url = st.text_input("Enter a YouTube or Website URL:")

# ✅ Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
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

                # ✅ Load content from YouTube or Website
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        yt_loader = YoutubeLoader.from_youtube_url(
                            generic_url, 
                            add_video_info=False,
                            language=["hi"]  # ✅ Change if another language is needed
                        )
                        transcript = yt_loader.load()

                        if transcript:
                            docs = transcript
                        else:
                            st.error("⚠️ No transcript available for this YouTube video.")
                            docs = None  # Avoid invalid API calls

                    except Exception as yt_error:
                        st.error(f"⚠️ Failed to fetch transcript: {yt_error}")
                        docs = None

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
