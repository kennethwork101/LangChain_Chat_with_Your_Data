""" 
The TF-IDF Retriever is an information retrieval technique that can be used to search through a dataset to find the most relevant data for the next stage of the model
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection or corpus 1. 
It is calculated by multiplying the term frequency (TF) of a word in a document by the inverse document frequency (IDF) of the word across a collection of documents 1. 
The TF component measures how frequently a word appears in a document, while the IDF component measures how rare the word is across the entire corpus 12.
The TF-IDF score of a word increases proportionally to the number of times it appears in the document and is offset by the frequency of the word in the corpus 1. 
This means that words that appear frequently in a document but also frequently in the corpus will have a lower TF-IDF score than words that appear frequently in a document but rarely in the corpus 1.
TF-IDF is commonly used in information retrieval and text mining as a weighting factor for scoring the relevance of a document to a query 12. 
It is used to rank documents based on their relevance to a query, with documents that have a higher TF-IDF score being considered more relevant 1.
If you are interested in building your own question answering system, you can use the TF-IDF Retriever, which is an agent that 
constructs a TF-IDF matrix for all entries in a given task. It generates responses by returning the highest-scoring documents for a query
"""

import argparse

from kwwutils import (
    clock,
    execute,
    format_docs,
    get_documents_by_path,
    get_embeddings,
    printit,
)
from langchain_community.retrievers import TFIDFRetriever

_path = "../../"


@clock
@execute
def main(options):
    embeddings = get_embeddings(options)
    printit("1 embeddings", embeddings)

    # Retrieve a list of documents
    docs = get_documents_by_path(options['filename'])
    printit("2 docs", docs)


    '''
    # Retrieve texts from documents, including new lines
    texts = [doc.page_content for doc in docs]
    printit("3 texts 1", texts)

    # Remove new lines, list of list
    texts = [text.split("\n") for text in texts]
    printit("4 texts 2", texts)

    # Turn texts into one list of strings
    texts = [" ".join(text) for text in texts]
    printit("5 texts 3", texts)
    '''

    # Call format_doc is similar to the above commented out lines
    texts = format_docs(docs)

    question = options["question"]
    tfidf_retriever = TFIDFRetriever.from_texts([texts])   
    response = tfidf_retriever.get_relevant_documents(question)
    printit(question, response)


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedding: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--filename', type=str, help='filename: ', default='../../data/data_all/pdf_files/kwong_resume_3.pdf')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--question', type=str, help='question', default='What is RabbitMQ and Kombu?')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="llama2")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument('--models', nargs='+', default=[
        "codellama:7b",        
        "codellama:7b-python",        
        "codellama:13b",        
        "codellama:13b-python",        
        "codellama:34b",        
        "codellama:34b-python",        
        "llama2:latest",           
        "llama2-uncensored:latest",           
        "medllama2:latest",        
        "medllama2:latest",        
        "mistral:instruct",        
        "mistrallite:latest",      
        "openchat:latest",         
        "orca-mini:latest",        
        "phi:latest",        
        "vicuna:latest",           
        "wizardcoder:latest",
        "wizardlm-uncensored:latest",        
        "yarn-llama2:latest",        
        "yarn-mistral:latest",        
    ])
    return vars(parser.parse_args())


if __name__ == '__main__':
    options = Options()
    main(**options)