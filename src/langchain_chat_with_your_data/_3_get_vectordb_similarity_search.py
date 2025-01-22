import argparse

from kwwutils import clock, execute, get_vectordb, printit

_path = "../../"


@clock
@execute
def main(options):
    vectordb = get_vectordb(options)
    question = options["question"]
    response = vectordb.similarity_search(question, k=3)
    printit("vectordb", vectordb)
    printit("vectordb cnt:", vectordb._collection.count())
    printit(question, response)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persist_directory",
        type=str,
        help="persist_directory",
        default=f"{_path}mydb/data_all/",
    )
    """ 
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='huggingface')
    """
    parser.add_argument(
        "--embedding",
        type=str,
        help="embedding: chroma gpt4all huggingface",
        default="chroma",
    )
    parser.add_argument(
        "--embedmodel", type=str, help="embedding: ", default="all-MiniLM-L6-v2"
    )
    #   parser.add_argument('--question', type=str, help='question', default='What did they say about matlab?')
    parser.add_argument(
        "--question",
        type=str,
        help="question",
        default="What did they say about fossil fuel projects?",
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--model", type=str, help="model", default="llama2")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "codebooga:latest",
            "codellama:13b",
            "codellama:13b-python",
            "codellama:34b",
            "codellama:34b-python",
            "codellama:7b",
            "codellama:7b-python",
            "codeup:latest",
            "deepseek-coder:latest",
            "dolphin-mistral:latest",
            "dolphin-mixtral:latest",
            "falcon:latest",
            "llama-pro:latest",
            "llama2-uncensored:latest",
            "llama2:latest",
            "magicoder:latest",
            "meditron:latest",
            "medllama2:latest",
            "mistral-openorca:latest",
            "mistral:instruct",
            "mistral:latest",
            "mistrallite:latest",
            "mixtral:latest",
            "openchat:latest",
            "orca-mini:latest",
            "orca2:latest",
            "phi:latest",
            "phind-codellama:latest",
            "sqlcoder:latest",
            "stable-code:latest",
            #       "starcoder:latest",
            #       "starling-lm:latest",
            #       "tinyllama:latest",
            "vicuna:latest",
            #       "wizardcoder:latest",
            #       "wizardlm-uncensored:latest",
            #       "yarn-llama2:latest",
            #       "yarn-mistral:latest",
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)
