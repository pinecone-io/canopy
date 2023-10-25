import pytest

from canopy.knowledge_base.chunker import MarkdownChunker
from canopy.knowledge_base.models import KBDocChunk
from canopy.models.data_models import Document
from tests.unit.chunker.base_test_chunker import BaseTestChunker


class TestMarkdownChunker(BaseTestChunker):

    @staticmethod
    @pytest.fixture(scope="class")
    def chunker():
        return MarkdownChunker(chunk_size=56,
                               chunk_overlap=10)

    @staticmethod
    @pytest.fixture
    def text():
        long_text = ("In a sleepy village, Lay a hidden secret. Anna found a map, "
                     "Old, torn, and creased. It marked a spot, Beneath the old oak. "
                     "With shovel in hand, Anna began to dig. Hours turned to minutes, "
                     "Then, a chest appeared. Golden light spilled out, The village's "
                     "lost lore. Inside, not gold, But memories and tales. Of brave "
                     "ancestors, And magical whales.")
        return f"""# Markdown Example for Unit Testing

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

## Another second level header
text after second level header

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

## long text
{long_text}

Inside, not gold, But memories and tales.
Of brave ancestors, And magical whales.

Anna shared the stories, Under stars so bright.
The village united, Bathed in tales' light.

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
                metadata={"test": 1},
                source="doc_1"),
            Document(
                id="test_document_3",
                text="# short markdown\nmarkdown is short",
                metadata={"test": 2},
            )
        ]

    @staticmethod
    @pytest.fixture
    def expected_chunks(documents):
        chunks = [
            KBDocChunk(
                id='test_document_1_0',
                text='# Markdown Example for Unit Testing\n\n## Headers\n### Level 3'
                     '\ntext in level 3\n#### Level 4\ntext in level 4\n##### Level 5'
                     '\ntext in level 5\n###### Level 6\ntext in level 6',
                source='doc_1',
                metadata={'test': '1'},
                document_id='test_document_1'),

            KBDocChunk(
                id='test_document_1_1',
                text='## Emphasis\n\n*Italic text* or _Italic text_'
                     '\n\n**Bold text** or '
                     '__Bold text__'
                     '\n\n**_Bold and italic_** or *__Bold and italic__*\n\n'
                     '~~Strikethrough text~~\n\n'
                     '## Another second level header\ntext after second level header',
                     source='doc_1',
                     metadata={'test': '1'},
                     document_id='test_document_1'),

            KBDocChunk(
                id='test_document_1_2',
                text='## Another second level header\ntext after second level header'
                     '\n\n## Lists'
                     '\n\n### Unordered\n\n* Item 1\n* Item 2\n  * Sub-item 2.1'
                     '\n  * Sub-item 2.2'
                     '\n\n### Ordered\n\n1. First item\n2. Second item'
                     '\n   1. Sub-item 2.1\n   '
                     '2. Sub-item 2.2\n\n## Links\n\n[OpenAI](https://www.openai.com/)'
                     '\n\n## Images'
                     '\n\n![Alt text](https://www.example.com/image.jpg)'
                     '\n\n## Blockquotes\n\n'
                     '> This is a blockquote.',
                source='doc_1',
                metadata={'test': '1'},
                document_id='test_document_1'),

            KBDocChunk(id='test_document_1_3',
                       text='## long text',
                       source='doc_1',
                       metadata={'test': '1'},
                       document_id='test_document_1'),

            KBDocChunk(id='test_document_1_4',
                       text='In a sleepy village, Lay a hidden secret. '
                            'Anna found a map, Old, torn, and '
                            'creased. It marked a spot, Beneath the old oak. '
                            'With shovel in hand, Anna '
                            'began to dig. Hours turned to minutes, '
                            'Then, a chest appeared. Golden light '
                            'spilled out, The village\'s lost lore. '
                            'Inside, not gold, But memories and '
                            'tales. Of',
                       source='doc_1',
                       metadata={'test': '1'},
                       document_id='test_document_1'),

            KBDocChunk(id='test_document_1_5',
                       text='lost lore. Inside, not gold, '
                            'But memories and tales. '
                            'Of brave ancestors, And '
                            'magical whales.',
                       source='doc_1',
                       metadata={'test': '1'},
                       document_id='test_document_1'),

            KBDocChunk(id='test_document_1_6',
                       text="Inside, not gold, But memories and tales."
                            "\nOf brave ancestors, And magical "
                            "whales.\n\nAnna shared the stories, Under stars so bright."
                            "\nThe village united, "
                            "Bathed in tales' light.",
                       source='doc_1',
                       metadata={'test': '1'},
                       document_id='test_document_1'),

            KBDocChunk(id='test_document_1_7',
                       text="## Inline code\n\nHere's some inline `code`."
                            "\n\n## Code blocks\n\n"
                            "```python\ndef hello_world():"
                            "\n    print(\"Hello, world!\")"
                            "\n```\n## table"
                            "\na | b | c\n--- | --- | ---\n1 | 2 | 3",
                       source='doc_1',
                       metadata={'test': '1'},
                       document_id='test_document_1'),

            KBDocChunk(id='test_document_3_0',
                       text='# short markdown\nmarkdown is short',
                       source='',
                       metadata={'test': '2'},
                       document_id='test_document_3')
        ]
        return chunks
