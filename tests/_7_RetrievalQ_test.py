import pytest
from kwwutils import clock, printit

from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._7_RetrievalQ_needwork import (
    main,
)


@clock
@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "map_rerank", "refine"])
def test_func(options, model, chain_type):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "llm"
    options["chain_type"] = chain_type
    options["question"] = "What did they say about fossil fuel projects?"
    response = main(**options)
    printit(f"{model} {chain_type} response", response)
    printit(f"{model} {chain_type} response type", type(response))
    printit(f"{model} {chain_type} response['result']", response["result"])
    printit(f"{model} {chain_type} question", options["question"])
    sources = [r.metadata['source'] for r in response["source_documents"]]
    checks = all([s.endswith("result.txt") for s in sources])
    printit("sources", sources)
    printit("checks", checks)
    assert checks