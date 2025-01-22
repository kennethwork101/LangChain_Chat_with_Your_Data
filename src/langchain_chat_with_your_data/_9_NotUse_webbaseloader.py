import argparse

from kwwutils import clock, execute, get_documents_by_path, printit

_path = "../../"


@clock
@execute
def main(options):
    url = options["url"]
    docs = get_documents_by_path(url)
    printit(url, docs)
    return docs


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
        ],
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)
