import pytest
from kwwutils import clock, printit

from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._4_ContextualCompressionRetriever import (
    main,
)


@clock
@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "map_rerank", "refine"])
def test_func(options, model, chain_type):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["chain_type"] = chain_type
    options["llm_type"] = "llm"
    options["question"] = "What did they say about fossil fuel projects?"
    response = main(**options)
    printit("response", response)
    printit("question", options["question"])
    sources = [r.metadata["source"] for r in response]
    checks = all([s.endswith("result.txt") for s in sources])
    printit("checks", checks)
    assert checks
