import argparse

from kwwutils import clock, execute, get_llm, get_vectordb, printit
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

_path = "../../"


@clock
@execute
def main(options):
    llm = get_llm(options)
    vectordb = get_vectordb(options)
    question = options["question"]
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever(search_type="mmr")
    )
    response = compression_retriever.get_relevant_documents(question)
    printit(question, response)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedding: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='llm')
    parser.add_argument('--question', type=str, help='question', default='What did they say about fossil fuel projects?')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="llama2:latest")
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