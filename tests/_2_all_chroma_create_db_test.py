import pytest
from kwwutils import clock, printit

from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._2_allchroma_create_db import (
    main,
)


@pytest.mark.vectordball
@pytest.mark.vectordbfew
@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["question"] = "What did they say about fossil fuel projects?"
    results = main(**options)
    result = [
        (model, embedding, vectordb_cnt, len(docs["documents"]) == vectordb_cnt)
        for model, docs, vectordb_cnt, embedding in results
    ]
    printit("result", result)
