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
- `num_outputs` - Sets the maximum number of characters that the model will generate in its response.
- `max_chunk_overlap` - This parameter is designed to make the application outputs flow better and make sense. It takes the last 20 characters from the previously generated chunk of words and looks to connect it with the next chunk of words.
- `chunk_size_limit` - Sets the maximum size of a single chunk (generated segment of text that can be combined with other chunks to create a piece of writing that makes sense) of text that the model can generate
- `prompt_helper` - This is an object, which uses the `PromptHelper` class to constrain the users input text to the parameters above. This class helps format input prompts for the OpenAI API client.
- `llm_predictor` - This creates a new object, which acts as a wrapper around the OpenAI API client and allows the use of their LLMs for text generation. The `LLMPredictor` class is also used to integrate GPT-3.5 with LlamaIndex.
    - `llm` - Specifies the LLM model to use for text generation.
    - `temperature` - Specifies the randomness of the generated text and ranges from 0-1. A temperature of 0.5 means that the model will generate text that is somewhat diverse, but still coherent and relevant to the context.
    - `model_name` - Specifies the LLM model to use.
    - `openai_api_key` - Provides OpenAI with your key for accessing their API.
- `service_context` -  This object creates a new python dataclass, that is built using the previously explained components (`llm_predictor` and `prompt_helper`)


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
The `def data_ingestion_indexing()` is a function that is responsible for loading data and then creating an index of that data so that it can be queried. Below is a further breakdown of the code:
- `documents` - This object leverages the `llama_index` `SimpleDirectoryReader`module to read the contents of a directory and return a list of documents (strings) that represents the contents of each file.
- `index` - This object creates a new instance of the `GPTVectorStoreIndex` class using a list of documents returned by `SimpleDirectoryReader`. This means that it's taking our `documents`, creating vector representations, and storing them for future querying. Additionally, it passes in the `create_service_context()` so that the `service_context` is properly configured, as it will be used by the index.
- `index.storage_context.persist()` - This line saves the index to the location described in the code below. The `persist` method writes the index data to that location.
- `return index` - This line returns the newly created index object so that it can be used later for querying.
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
The `def data_querying()` is a function that takes in the users input text, uses the `storage_context` to retrieve an index, and then uses the index to search for an answer to the users input.
- `storage_context` - This object creates a new instance of a class called `StorageContext` and stores the data in a directory ("./storage"). Basically it uses default module settings and creates a container to hold the data.
- `index` - This object uses the `load_index_from_storage` function to load our index into our storage container. Additionally, we include the `service_context` argument so that we can interact with the storage system.
- `response` - This line uses the index to search for the users input text. The `as_query_engine()` method turns the index into a query engine to answer questions. The `query()` method takes the input text and searches the index for a response.
- `return response.response` - This line returns the result of the users input query.

```python
iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your question"),
                     outputs="text",
                     title="ARS Q and A")
```
The `iface` object creates a new GUI interface using `gradio` and titles the interface "ARS Q and A". 
- `fn` - This explains the function that will be called when the user clicks the "Query" button.
- `inputs` - This explains the list of input fields the user can fill out (in this case it's just a text box with 7 lines that's labeled "Enter your question").
- `outputs` - The type of output expected from the `fn` function, which is a string value represented by "text".

```python
#passes in data directory
index = data_ingestion_indexing("data")
iface.launch(share=False)
```
