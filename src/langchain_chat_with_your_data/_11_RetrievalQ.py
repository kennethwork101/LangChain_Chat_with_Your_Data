import argparse

from kwwutils import clock, execute, get_llm, get_vectordb, printit
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

_path = "../../"

template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. Use three sentences maximum. 
Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:
"""


@clock
@execute
def main(options):
    llm = get_llm(options)
    vectordb = get_vectordb(options)
    retriever = vectordb.as_retriever()
    prompt = PromptTemplate.from_template(template=template)
    question = options["question"]
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
    )
    response = qa_chain.invoke({"query": question})
    printit("PromptTemplate", prompt)
    printit("qa_chain", qa_chain)
    printit("qa_chain type", type(qa_chain))
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
        "--question",
        type=str,
        help="question",
        default="What did they say about fossil fuel projects?",
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="llama2")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "codellama:7b",
            "llama2:latest",
            "medllama2:latest",
            "mistral:instruct",
            "mistrallite:latest",
            "openchat:latest",
            "orca-mini:latest",
            "vicuna:latest",
            "wizardcoder:latest",
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)
