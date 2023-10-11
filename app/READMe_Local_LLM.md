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

```python
def create_service_context():
    max_input_size = 4096
    num_outputs = 512
    prompt_helper = PromptHelper(max_input_size, num_outputs,  chunk_overlap_ratio=0.2)
    llm_predictor = LLMPredictor(llm=CustomLLM())
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return service_context
```

```python
def data_ingestion_indexing(directory_path):
    documents = SimpleDirectoryReader(directory_path, filename_as_id=True).load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=create_service_context()
    )
    index.storage_context.persist()
    return index
```

```python
storage_context = StorageContext.from_defaults(persist_dir="C:\\Users\\M30880\\OneDrive - Noblis\\Documents\\AI Explorers\\R&D\\storage")
index = load_index_from_storage(storage_context, service_context=create_service_context())
```

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

```python
def query_index(input_text):
    response = index.as_query_engine().query(input_text)
    doc_id = response.get_formatted_sources()
    return {"response": response.response, "information_origin": doc_id}
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











