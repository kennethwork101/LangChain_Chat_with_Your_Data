from kwwutils import clock, printit
from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._6_ContextualCompressionRetriever_mmr import (
    main,
)
import pytest


@pytest.mark.skip
@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "llm"
    options["question"] = "What did they say about fossil fuel projects?"
    response = main(**options)
    printit("response", response)
    printit("question", options["question"])
    printit("response len", len(response))
    printit("response metadata", response[0].metadata)
    output = response[0].page_content.lower()
    assert "china" in output
