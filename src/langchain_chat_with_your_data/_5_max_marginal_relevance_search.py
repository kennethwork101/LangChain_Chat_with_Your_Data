""" 
The idea behind using MMR is that it tries to reduce redundancy and increase diversity in the result and 
is used in text summarization. MMR selects the phrase in the final keyphrases list according to 
a combined criterion of query relevance and novelty of information.
"""

import argparse

from kwwutils import clock, execute, get_vectordb, printit

_path = "../../"


@clock
@execute
def main(options):
    vectordb = get_vectordb(options)
    question = options["question"]
    response = vectordb.max_marginal_relevance_search(question, k=2, fetch_k=3)
    printit(question, response)
    printit("Document", type(response[0]))
    printit("Document .to_json()", response[0].to_json())
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedding: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--question', type=str, help='question', default='What did they say about matlab?')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
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