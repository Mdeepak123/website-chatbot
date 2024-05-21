from dotenv import load_dotenv
import os
from pinecone import Pinecone
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.chat_engine.types import ChatMode
from llama_index import download_loader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
import streamlit as st
from node_postprocessors import duplicate_postprocessors

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ['PINECONE_ENVIRONMENT'] )

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    #initialize pinecone
    index_name = "website-db"
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )

index = get_index()
#initialize chat engine
if "chat-engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5, #keep top 50% of related chunks
        threshold_cutoff=0.7, #leave sentances with similarity score above .7
    )

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True,
        node_postprocessors=[postprocessor, duplicate_postprocessors.DuplicateRemoverNodePostprocessor()],
    )

st.set_page_config(
    page_title="Chat with your website",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with your website")

#adding the past messages in a session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about your website",
        }
    ]

#user input
if prompt := st.chat_input("Your Question: "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    

#display past msgs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#get response from ai
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)