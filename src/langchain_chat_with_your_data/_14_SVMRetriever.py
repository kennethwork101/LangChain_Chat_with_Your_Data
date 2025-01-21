import argparse

from langchain_community.retrievers import SVMRetriever

_path = "../../"

from kwwutils import (
    clock,
    execute,
    format_docs,
    get_documents_by_path,
    get_embeddings,
    printit,
)


@clock
@execute
def main(options):
    embeddings = get_embeddings(options)
    docs = get_documents_by_path(options['filename'])
    doc_str = format_docs(docs)
    question = options["question"]
    # Must put doc_str in a list
    svm_retriever = SVMRetriever.from_texts([doc_str], embeddings)
    response = svm_retriever.get_relevant_documents(question)
    printit("embeddings", embeddings)
    printit("docs", docs)
    printit("doc_str", doc_str)
    printit("doc_str len", len(doc_str))
    printit("doc_str type", type(doc_str))
    printit(question, response)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedding: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--filename', type=str, help='filename: ', default='../../data/data_all/pdf_files/kwong_resume_3.pdf')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='llm')
#   parser.add_argument('--question', type=str, help='question', default='What did they say about matlab?')
    parser.add_argument('--question', type=str, help='question', default='What did they say about Kenneth?')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="codellama:7b")
    return vars(parser.parse_args())


if __name__ == '__main__':
    options = Options()
    main(**options)