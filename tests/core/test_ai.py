from langchain.chat_models.base import BaseChatModel
from langchain_community.chat_models.fake import FakeListChatModel

from gpt_engineer.core.ai import AI, HumanMessage, AIMessage, SystemMessage


def mock_create_chat_model(self) -> BaseChatModel:
    return FakeListChatModel(responses=["response1", "response2", "response3"])


def test_start(monkeypatch):
    monkeypatch.setattr(AI, "_create_chat_model", mock_create_chat_model)

    ai = AI("gpt-4")

    # act
    response_messages = ai.start("system prompt", "user prompt", "step name")

    # assert
    assert response_messages[-1].content == "response1"


def test_next(monkeypatch):
    # arrange
    monkeypatch.setattr(AI, "_create_chat_model", mock_create_chat_model)

    ai = AI("gpt-4")
    response_messages = ai.start("system prompt", "user prompt", "step name")

    # act
    response_messages = ai.next(
        response_messages, "next user prompt", step_name="step name"
    )

    # assert
    assert response_messages[-1].content == "response2"


def test_token_logging(monkeypatch):
    # arrange
    monkeypatch.setattr(AI, "_create_chat_model", mock_create_chat_model)

    ai = AI("gpt-4")

    # act
    response_messages = ai.start("system prompt", "user prompt", "step name")
    usageCostAfterStart = ai.token_usage_log.usage_cost()
    ai.next(response_messages, "next user prompt", step_name="step name")
    usageCostAfterNext = ai.token_usage_log.usage_cost()

    # assert
    assert usageCostAfterStart > 0
    assert usageCostAfterNext > usageCostAfterStart

def test_remove_continue_messages(monkeypatch):
    monkeypatch.setattr(AI, "_create_chat_model", mock_create_chat_model)
    # arrange
    ai = AI("gemini-pro")
    messages = [
        HumanMessage(content="continue: your text that was not finished"),
        HumanMessage(content="continue: your text that was not finished"),
        HumanMessage(content="continue: your text that was not finished"),
    ]

    # act
    messages = ai._remove_continue_messages(messages=messages, target="continue: your text that was not finished")

    # assert
    assert len(messages) == 0

    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="continue: your text that was not finished"),
        SystemMessage(content="message"),
        AIMessage(content="AI"),
        HumanMessage(content="continue: your text that was not finished"),
        AIMessage(content="message"),
    ]

    result = [
        SystemMessage(content="system"),
        SystemMessage(content="message"),
        AIMessage(content="AI"),
        AIMessage(content="message"),
    ]

    new_messages = ai._remove_continue_messages(messages=messages, target="continue: your text that was not finished")
    assert new_messages == result

def test_merge_continue_messages_with_block_not_completed(monkeypatch):
    monkeypatch.setattr(AI, "_create_chat_model", mock_create_chat_model)
    ai = AI("gemini-pro")

    messages = [
        AIMessage(content=
"""
path/demo.py
```
def test:
```"""),
        AIMessage(content=
"""
    print("message")
```
"""),
    ]

    result_content = """
path/demo.py
```
def test:
    print("message")
```
"""
    new_messages = ai._merge_message(messages=messages)
    assert new_messages[0].content == result_content


    messages = [
        AIMessage(content=
"""
path/demo.py
```
def test:
```
"""),
        AIMessage(content=
"""
```
    print("message")
```
"""),
    ]

    result_content = """
path/demo.py
```
def test:
```

```
    print("message")
```
"""
    new_messages = ai._merge_message(messages=messages)
    assert new_messages[0].content == result_content
