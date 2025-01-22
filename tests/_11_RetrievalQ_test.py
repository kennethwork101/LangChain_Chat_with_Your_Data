from kwwutils import clock, printit
from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._11_RetrievalQ import (
    main,
)


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
    result = response["result"].lower()
    sources = [r.metadata["source"] for r in response["source_documents"]]
    printit("sources", sources)
    checks = all([s.endswith("result.txt") for s in sources])
    assert response["query"] == options["question"]
    assert "fossil" in result
    assert "fuel" in result
    assert checks
