import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
import pickle 

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL (YouTube or Website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")

## Initialize the Groq Model (Gemma2-9b)
if groq_api_key.strip():
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
else:
    llm = None

## Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

## Function to Extract YouTube Transcript
def get_youtube_transcript(video_url):
    try:
        # Extract YouTube Video ID
        video_id = video_url.split("v=")[-1] if "v=" in video_url else video_url.split("/")[-1]
        
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([t["text"] for t in transcript])
        return transcript_text
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

## Process When Button Clicked
if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube video or website).")
    else:
        try:
            with st.spinner("Processing..."):
                ## Load YouTube or Website Content
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    transcript_text = get_youtube_transcript(generic_url)
                    if "Error" in transcript_text:
                        st.error(f"Failed to fetch YouTube transcript: {transcript_text}")
                        docs = []
                    else:
                        docs = [Document(page_content=transcript_text)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()

                ## Ensure content is available before summarization
                if not docs:
                    st.error("No content found to summarize.")
                else:
                    text_content = "\n\n".join([doc.page_content for doc in docs])

                    ## Summarization Chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.invoke({"input_documents": [Document(page_content=text_content)]})

                    ## Extract summary text
                    summary_text = output_summary.get("output_text", "No summary generated.")

                    st.success(summary_text)

        except Exception as e:
            st.error(f"Exception: {str(e)}") 

with open("app.pkl", "wb") as file:
    pickle.dump(llm, file)
print("Model saved as app.pkl")

with open("app.pkl", "rb") as file:
    loaded_model = pickle.load(file)
