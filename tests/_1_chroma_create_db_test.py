from kwwutils import clock, printit

from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._1_chroma_create_db import main


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["question"] = "What did they say about fossil fuel projects?"
    docs, vectordb_cnt = main(**options)
    printit("vectordb_cnt", vectordb_cnt)
    printit("docs", len(docs))
    assert len(docs["documents"]) == vectordb_cnt, f"-Error: vectordb_cnt {vectordb_cnt} != {len(docs['documents'])}"