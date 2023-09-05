import pytest
from context_engine.knoweldge_base.chunker.recursive_character \
    import RecursiveCharacterChunker
from context_engine.knoweldge_base.models import KBDocChunk
from context_engine.models.data_models import Document
from tests.unit.chunker.base_test_chunker import BaseTestChunker
from tests.unit.stubs.stub_tokenizer import StubTokenizer


class TestMarkdownChunker(BaseTestChunker):

    @staticmethod
    @pytest.fixture(scope="class")
    def chunker():
        return RecursiveCharacterChunker(chunk_size=40,
                                         chunk_overlap=30,
                                         tokenizer=StubTokenizer())

    @staticmethod
    @pytest.fixture
    def text():
        return """# Markdown Example for Unit Testing

## Headers
### Level 3
text in level 3
#### Level 4
text in level 4
##### Level 5
text in level 5
###### Level 6
text in level 6

## Emphasis

*Italic text* or _Italic text_

**Bold text** or __Bold text__

**_Bold and italic_** or *__Bold and italic__*

~~Strikethrough text~~

# Another first level header
text after first level header

## Lists

### Unordered

* Item 1
* Item 2
  * Sub-item 2.1
  * Sub-item 2.2

### Ordered

1. First item
2. Second item
   1. Sub-item 2.1
   2. Sub-item 2.2

## Links

[OpenAI](https://www.openai.com/)

## Images

![Alt text](https://www.example.com/image.jpg)

## Blockquotes

> This is a blockquote.

## Inline code

Here's some inline `code`.

## Code blocks

```python
def hello_world():
    print("Hello, world!")
```
## table
a | b | c
--- | --- | ---
1 | 2 | 3
"""

    @staticmethod
    @pytest.fixture
    def documents(text):
        return [
            Document(
                id="test_document_1",
                text=text,
                metadata={"test": 1}),
            Document(
                id="test_document_3",
                text="# short markdown\nmarkdown is short",
                metadata={"test": 2},
            )
        ]

    @staticmethod
    @pytest.fixture
    def expected_chunks(documents):
        return [
            KBDocChunk(id='test_document_1_0',
                       text='# Markdown Example for Unit Testing\n\n## Headers\n### Level 3\ntext in level 3\n#### Level 4\ntext in level 4\n##### Level 5\ntext in level 5\n###### Level 6\ntext in level 6\n\n## Emphasis',  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_1',
                       text='## Emphasis\n\n*Italic text* or _Italic text_\n\n**Bold text** or __Bold text__\n\n**_Bold and italic_** or *__Bold and italic__*\n\n~~Strikethrough text~~\n\n# Another first level header\ntext after first level header\n\n## Lists\n\n### Unordered',  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_2',
                       text='**Bold text** or __Bold text__\n\n**_Bold and italic_** or *__Bold and italic__*\n\n~~Strikethrough text~~\n\n# Another first level header\ntext after first level header\n\n## Lists\n\n### Unordered\n\n* Item 1\n* Item 2\n  * Sub-item 2.1\n  * Sub-item 2.2',  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_3',
                       text='~~Strikethrough text~~\n\n# Another first level header\ntext after first level header\n\n## Lists\n\n### Unordered\n\n* Item 1\n* Item 2\n  * Sub-item 2.1\n  * Sub-item 2.2\n\n### Ordered',  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_4',
                       text='# Another first level header\ntext after first level header\n\n## Lists\n\n### Unordered\n\n* Item 1\n* Item 2\n  * Sub-item 2.1\n  * Sub-item 2.2\n\n### Ordered\n\n1. First item\n2. Second item\n   1. Sub-item 2.1\n   2. Sub-item 2.2',  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_5',
                       text='## Lists\n\n### Unordered\n\n* Item 1\n* Item 2\n  * Sub-item 2.1\n  * Sub-item 2.2\n\n### Ordered\n\n1. First item\n2. Second item\n   1. Sub-item 2.1\n   2. Sub-item 2.2\n\n## Links\n\n[OpenAI](https://www.openai.com/)\n\n## Images\n\n![Alt text](https://www.example.com/image.jpg)\n\n## Blockquotes',  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_6',
                       text="### Ordered\n\n1. First item\n2. Second item\n   1. Sub-item 2.1\n   2. Sub-item 2.2\n\n## Links\n\n[OpenAI](https://www.openai.com/)\n\n## Images\n\n![Alt text](https://www.example.com/image.jpg)\n\n## Blockquotes\n\n> This is a blockquote.\n\n## Inline code\n\nHere's some inline `code`.\n\n## Code blocks",  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_1_7',
                       text='## Blockquotes\n\n> This is a blockquote.\n\n## Inline code\n\nHere\'s some inline `code`.\n\n## Code blocks\n\n```python\ndef hello_world():\n    print("Hello, world!")\n```\n## table\na | b | c\n--- | --- | ---\n1 | 2 | 3',  # noqa: E501
                       metadata={'test': '1'},
                       document_id='test_document_1'),
            KBDocChunk(id='test_document_3_0',
                       text='# short markdown\nmarkdown is short',
                       metadata={'test': '2'},
                       document_id='test_document_3')]
