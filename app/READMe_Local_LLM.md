# PDF_Chatbot

## Overview
This is the code documentation for a chatbot that is designed to provide quick and efficient responses by querying a pre-defined data index. The chatbot is built on top of the LangChain's LLMChain and LlamaIndex infrastructure, with integration to a Local Large Language Model for query handling.

## Installation
`pip install -r requirements.txt`

## Running the application
Be sure to re-direct the following two fields:
`storage_context` - change the path to where you want the index stored
`data_ingestion_indexing` - change the path to where you have stored the files you want indexed

Run the application using the code below in the terminal (make sure the directory is set to where the application is located):

`python app.py`

## Code Information
```python
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-max-1.3b")
```
This line loads a pretrained tokenizer (`AutoTokenizer`) from the HuggingFace transformers library. The tokenizer will convert text into a format that the model can understand.
```python
class CustomLLM(LLM):
    model_name = OPTForCausalLM.from_pretrained("facebook/opt-iml-max-1.3b")
    pipeline = pipeline("text-generation", model=model_name, tokenizer=tokenizer)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=512)[0]["generated_text"]
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
```
This `class Custom(LLM)` section defines a custom Large Language Model (LLM). The class (`CustomLLM`) inherits functionalities from the `LLM` base class. Below is a further breakdown of the code:
- `model_name` - This line initializes the pretrained LLM ("facebook/opt-iml-max-1.3b"). It leverages the `OPTForCausalLM` model architecture from the HuggingFace Transformers library.
- `pipeline` - This line creates a text generation pipeline using the locally-loaded model and its corresponding tokenizer.
- `def _call` - This method takes a text prompt, sends it to the language model for generation, and returns the generated text without including the original prompt. The model's `response` calls on the `pipeline` to generate the response, which is limited to 512 tokens.
- `def _identifying_params(self)` - This property provides identifying parameters for the custom LLM. It returns a dictionary with a single key-value pair.
- `def _llm_type(self)` - This property simply defines the LLM as "custom". Which could be useful if we tested other LLMs
```python
def create_service_context():
    max_input_size = 4096
    num_outputs = 512
    prompt_helper = PromptHelper(max_input_size, num_outputs,  chunk_overlap_ratio=0.2)
    llm_predictor = LLMPredictor(llm=CustomLLM())
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return service_context
```
The `def create_service_context()` is a function that creates and configures a container on the OpenAI platform. Below is a further breakdown of `create_service_context()` code:
- `max_input_size` - Sets the maximum size of the user input text that the model will process, before the model shortens it.
- `num_outputs` - Sets the maximum number of characters that the model will generate in its response.
- `prompt_helper` - This is an object, which uses the `PromptHelper` class to constrain the users input text to the parameters above. Additionally, it incorporates the `chunk_overlap_ratio` which defines how much the outputs overlap when chunking the data. This class helps format input prompts for the local LLM.
- `llm_predictor` - This creates a predictor object for the custom LLM. `LLMPredictor` acts as a wrapper around the `CustomLLM`.  
- `service_context` -  This object creates a new python dataclass, that is built using the previously explained components (`llm_predictor` and `prompt_helper`)
```python
def data_ingestion_indexing(directory_path):
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=create_service_context()
    )
    index.storage_context.persist()
    return index
```
The `def data_ingestion_indexing()` is a function that is responsible for loading data and then creating an index of that data so that it can be queried. Below is a further breakdown of the code:
- `documents` - This object leverages the `llama_index` `SimpleDirectoryReader` module to read the contents of a directory and return a list of documents (strings) that represents the contents of each file.
- `index` - This object creates a new instance of the `VectorStoreIndex` class using a list of documents returned by `SimpleDirectoryReader`. This means that it's taking our `documents`, creating vector representations, and storing them for future querying. Additionally, it passes in the `create_service_context()` so that the `service_context` is properly configured, as it will be used by the index.
- `index.storage_context.persist()` - This line saves the index to the location described in the code below. The `persist` method writes the index data to that location.
- `return index` - This line returns the newly created index object so that it can be used later for querying.
```python
storage_context = StorageContext.from_defaults(persist_dir="ADD_IN_PATH_TO_WHERE_INDEX_WILL_BE")
index = load_index_from_storage(storage_context, service_context=create_service_context())
```
- `storage_context` - This object creates a new instance of a class called `StorageContext` and stores the data in a directory. Basically it uses default module settings and creates a container to hold the data.
- `index` - This object uses the `load_index_from_storage` function to load our index into our storage container. Additionally, we include the `service_context` argument so that we can interact with the storage system.
```python
def data_querying(input_text):
    input_texts = [q.strip() for q in input_text.split("?") if q.strip()]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda q: query_index(q), input_texts))

    combined_results = "\n\n".join(
        [f"Q: {input_texts[i]}\nA: {res['response']}\nSource: {res['information_origin']}" for i, res in enumerate(results)]
    )
    
    return combined_results
```
The `def data_querying(input_text)` function splits user input into multiple questions and queries the index for each question. It then combines the results into a single output. Below is a further breakdown of the code:
- `input_texts` - This line takes the input string and splits it into individual questions based on a question mark "?".
- `ThreadPoolExecutor` - The code uses this to perform concurrent querying, meaning multiple questions can be queried at the same time to speed up the process.
- `combined_results` - The code takes the list of strings and combines them into a single string.
```python
def query_index(input_text):
    response = index.as_query_engine().query(input_text)
    doc_id = response.get_formatted_sources()
    return {"response": response.response, "information_origin": doc_id}
```
The `def query_index(input_text)` function directly interacts with the index to retrieve the response. Below is a further breakdown of the code:
- `response` - This line creates the query engine and searches the index for a response.
- `doc_id` - This line fetches the source or origin of the obtained response.
```python
iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your questions separated by a question mark"),
                     outputs="text",
                     title="Q and A App")
```
The `iface` object creates a new GUI interface using `gradio` and titles the interface "ARS Q and A". 
- `fn` - This explains the function that will be called when the user clicks the "Query" button.
- `inputs` - This explains the list of input fields the user can fill out (in this case it's just a text box with 7 lines that's labeled "Enter your question").
- `outputs` - The type of output expected from the `fn` function, which is a string value represented by "text".

```python
#passes in data directory
index = data_ingestion_indexing("ADD_IN_PATH_TO_DOCUMENTS_AND_DATA")
iface.launch(share=False)
```
`index` - This line loads and indexes data from a specific directory when the program starts.
`iface.launch` - This line launches the user interface and allows users to interact with it










