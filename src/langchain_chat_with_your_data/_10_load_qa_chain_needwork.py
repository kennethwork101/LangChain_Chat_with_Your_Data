"""
- need work on other chain_types accept for stuff
- Why template is not being used?
  Note only chain_type == stuff got valid output
"""

import argparse

from kwwutils import clock, execute, get_documents_by_path, get_llm, printit
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

_path = "../../"

template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:
"""

question_prompt_template = """
Use the following portion of a long document to see if any of the text is relevant 
to answer the question. 
{context}
Question: {question}
Relevant text, if any:
"""

combine_prompt_template = """
Given the following extracted parts of a long document and a question, 
create a final answer in the same language as the question. 

QUESTION: {question}
=========
{summaries}
=========
Answer in the original language of the question but answer with at most 1 sentences:
"""


@clock
@execute
def main(options):
    llm = get_llm(options)
    question = options["question"]
    chain_type = options["chain_type"]

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )
    path = options["pathname"]
    docs = get_documents_by_path(path)
    if chain_type == "stuff":
        qa_chain = load_qa_chain(
            llm,
            prompt=prompt,
            chain_type=chain_type,
        )
    else:
        qa_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            question_prompt=question_prompt,
            combine_prompt=combine_prompt,
        )
    response = qa_chain.invoke(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    printit(f"{chain_type}: {question}", response)
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
    parser.add_argument("--chain_type", type=str, help="chain_type", default="stuff")
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="chat"
    )
    parser.add_argument(
        "--pathname", type=str, help="pathname", default=f"{_path}data/data_all/"
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
