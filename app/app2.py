from llama_index import SimpleDirectoryReader, LLMPredictor, PromptHelper, StorageContext, ServiceContext, GPTVectorStoreIndex, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import openai

openai.api_key = "INSERT_API_KEY"

def create_service_context():
    #constraint parameters
    max_input_size = 4096
    num_outputs = 512
    #allows the user to explicitly set certain constraint parameters
    prompt_helper = PromptHelper(max_input_size, num_outputs,  chunk_overlap_ratio=0.2)
    #LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs, openai_api_key=openai.api_key))
    #constructs service_context
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return service_context

def data_ingestion_indexing(directory_path):
    #loads data from the specified directory path
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    
    #when first building the index
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=create_service_context()
    )
    #persist index to disk, default "storage" folder
    index.storage_context.persist()
    return index
    
def data_querying(input_text):
    #split the input text by question mark and strip any leading/trailing spaces
    input_texts = [q.strip() for q in input_text.split("?") if q.strip()]
    #rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="/workspaces/PDF_Chatbot/storage")
    #loads index from storage
    index = load_index_from_storage(storage_context, service_context=create_service_context())
    # queries the index with each input text
    results = []
    for input_text in input_texts:
        #queries the index with the input text
        response = index.as_query_engine().query(input_text)
        doc_id = response.get_formatted_sources()
        results.append({"response": response.response, "information_origin": doc_id})
    
    combined_results = "\n\n".join(
        [f"Q: {input_texts[i]}\nA: {res['response']}\nSource: {res['information_origin']}" for i, res in enumerate(results)]
    )
    return combined_results

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your questions separated by a question mark"),
                     outputs="text",
                     title="Q and A Bot")
#passes in data directory
index = data_ingestion_indexing("data")
iface.launch(share=False)
