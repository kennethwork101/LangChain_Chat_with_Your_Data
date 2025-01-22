import argparse

from kwwutils import clock, create_vectordb, execute, printit

_path = "../../"


@clock
@execute
def main(options):
    # Create vectordb based on a specific embedding and vectordb_type
    vectordb = create_vectordb(options)
    question = options["question"]
    response = vectordb.similarity_search(question, k=3)
    vectordb_cnt = vectordb._collection.count()
    printit("vectordb", vectordb)
    printit("vectordb cnt:", vectordb_cnt)
    printit(f"{options['model']}: {question}", response)
    # Get all documents from vectordb
    docs = vectordb.get()
    printit("doc['documents'] cnt", len(docs["documents"]))
    printit("doc keys", docs.keys())
    return docs, vectordb_cnt


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persist_directory",
        type=str,
        help="persist_directory",
        default=f"{_path}mydb/data_all/",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="embedding: chroma gpt4all huggingface",
        default="chroma",
    )
    parser.add_argument(
        "--embedmodel", type=str, help="embedmodel: ", default="all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="llm"
    )
    parser.add_argument(
        "--pathname", type=str, help="pathname", default=f"{_path}data/data_all/"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="question",
        default="What did they say about matlab?",
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument(
        "--vectordb_type", type=str, help="vectordb_type", default="disk"
    )
    parser.add_argument(
        "--vectorstore", type=str, help="vectorstore: Chroma, FAISS", default="Chroma"
    )
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
            "starcoder:latest",
            "starling-lm:latest",
            "tinyllama:latest",
            "vicuna:latest",
            "wizardcoder:latest",
            "wizardlm-uncensored:latest",
            "yarn-llama2:latest",
            "yarn-mistral:latest",
        ],
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)
