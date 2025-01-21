""" 
>>>-error<<<: 2 validation errors for LLMChain
"""
import argparse

import transformers
from kwwutils import clock, execute, get_documents_by_path, get_vectordb, printit
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline

_path = "../../"

@clock
@execute
def main(options):
    vectordb = get_vectordb(options)
    print(vectordb._collection.count())
    template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template,)
    print(f"prompt: {prompt}")
    """
    model = options['model']
    embeddings = get_embeddings(options)
    """
    model_name = 'Intel/neural-chat-7b-v3-1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained("my_model")
    printit("111 model type", type(model))
    printit("111 model", model)

    question = "Is probability a class topic?"
    response = model.generate(question)
    printit(f"111 question {question}", response)

    
    ''' 
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=vectordb.as_retriever(), 
        return_source_documents=True, 
        chain_type_kwargs={"prompt": prompt}
    )
    printit("111 qa_chain", qa_chain)
    response = qa_chain({"query": question})
    printit(question, response)
    '''


def use_conversation_retrievel(llm, prompt, vectordb, question, memory):
    print(f"1 memory {memory}")
    memory_maps = {
        "buffer_memory": ConversationBufferMemory,
    }
    retriever = vectordb.as_retriever()
    memory_fn = memory_maps[memory]
    memory = memory_fn(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa_chain({"question": question})
    print(f"222 conversation result: {result}")
    return result


def use_load_qa_chain(llm):
    question_prompt_template = """
    Use the following portion of a long document to see if any of the text is relevant 
    to answer the question. 
    Answer with at most 1 sentences:
    {context}
    Question: {question}
    Relevant text, if any:
    """
    QUESTION_PROMPT = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])

    combine_prompt_template = """
    Given the following extracted parts of a long document and a question, 
    create a final answer in the same language as the question. 
    Answer with at most 1 sentences:

    QUESTION: {question}
    =========
    {summaries}
    =========
    Answer in the original language of the question but answer with at most 1 sentences:
    """
    
    COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["summaries", "question"])
    docs = None
    path = "../../data/data_small/"
    docs = get_documents_by_path(path)
    print(f"XXX docs type {type(docs[0])}")
    query = "Is probability a class topic?"
    query = "Find info about automation or Python"
    qa_chain = load_qa_chain(llm, 
                       chain_type="map_reduce", 
                       question_prompt=QUESTION_PROMPT, 
                       combine_prompt=COMBINE_PROMPT)
    result = qa_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='huggingface')
    parser.add_argument('--embedmodel', type=str, help='embedding: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--model', type=str, help='model', default="llama2")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    return vars(parser.parse_args())


if __name__ == '__main__':
    options = Options()
    main(**options)
