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
