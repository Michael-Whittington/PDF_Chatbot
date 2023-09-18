# PDF_Chatbot

## Overview
This is the code documentation for a chatbot that is designed to provide quick and efficient responses by querying a pre-defined data index. The chatbot is built on top of the LangChain's LLMChain and LlamaIndex infrastructure, with integration to OpenAI's GPT models for query handling.

## Installation
`pip install -r requirements.txt`

## Running the application
Ensure that the OpenAI API key is set:
Replace `INSERT_API_KEY` in the code below:

`openai.api_key = "INSERT_API_KEY"`

Run the application using the code below in the terminal (make sure the directory is set to where the application is located):

`python app.py`

## Code Information
```python
def create_service_context():
    #constraint parameters
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    #allows the user to explicitly set certain constraint parameters
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    #LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs, openai_api_key=openai.api_key))
    #constructs service_context
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return service_context
```
The `def create_service_context()` is a function that creates and configures a container on the OpenAI platform. Below is a further breakdown of `create_service_context()` code:
- `max_input_size` - Sets the maximum size of the user input text that the model will process, before the model shortens it.
- `num_outputs`


```python
def data_ingestion_indexing(directory_path):
    #loads data from the specified directory path
    documents = SimpleDirectoryReader(directory_path).load_data()
    #when first building the index
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=create_service_context()
    )
    #persist index to disk, default "storage" folder
    index.storage_context.persist()
    return index
```

```python
def data_querying(input_text):
    #rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    #loads index from storage
    index = load_index_from_storage(storage_context, service_context=create_service_context())
    
    #queries the index with the input text
    response = index.as_query_engine().query(input_text)
    
    return response.response
```

```python
iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your question"),
                     outputs="text",
                     title="ARS Q and A")
```

```python
#passes in data directory
index = data_ingestion_indexing("data")
iface.launch(share=False)
```
