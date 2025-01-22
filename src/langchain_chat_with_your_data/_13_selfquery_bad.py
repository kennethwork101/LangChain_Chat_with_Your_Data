# ###   Got invalid JSON object. Error: Extra data: line 5 column 1 (char 175)
# All medels failed this test

import argparse

from kwwutils import clock, execute, get_llm, get_vectordb, printit
from langchain.chains.query_constructor.base import (
    AttributeInfo,
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.prompts import PromptTemplate

# from langchain.retrievers.self_query import ChromaTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator

_path = "../../"

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]


@clock
@execute
def main(options):
    llm = get_llm(options)
    vectordb = get_vectordb(options)
    document_content_description = "Lecture notes"
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )
    question = "what did they say about regression in the third lecture?"
    printit("question", question)
    response = retriever.get_relevant_documents(question)
    printit(question, response)

    # ### does not work: https://github.com/langchain-ai/langchain/issues/5882
    prompt = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
        schema_prompt=PromptTemplate(
            template="your rewritten prompt {allowed_comparators}, {allowed_operators}",
            input_variables=["allowed_comparators", "allowed_operators"],
        ),
    )
    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | llm | output_parser
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectordb,
        structured_query_translator=ChromaTranslator(),
    )
    response = retriever.invoke(
        "what did they say about regression in the third lecture?"
    )
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
        "--embedmodel", type=str, help="embedding: ", default="all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="llm"
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
            #       "codellama:7b",
            #       "llama2:latest",
            #       "medllama2:latest",
            #       "mistral:instruct",
            #       "mistrallite:latest",
            #       "openchat:latest",
            #       "orca-mini:latest",
            #       "vicuna:latest",
            "wizardcoder:latest",
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)
