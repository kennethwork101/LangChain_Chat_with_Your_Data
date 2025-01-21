import pytest
from kwwutils import clock, printit
from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._10_load_qa_chain_needwork import (
    main,
)


#@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "map_rerank", "refine"])
#@pytest.mark.parametrize("chain_type", ["stuff", "map_reduce"])
@pytest.mark.skip
@pytest.mark.parametrize("chain_type", ["stuff"])
@clock
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
    assert "output_text" in response
    result = response["output_text"].lower()
    assert "fossil" in result
    assert "fuel" in result
