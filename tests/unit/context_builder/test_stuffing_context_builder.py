from context_engine.context_engine.models import \
    ContextSnippet, ContextQueryResult
from context_engine.models.data_models import Context
from ..stubs.stub_tokenizer import StubTokenizer
from context_engine.knoweldge_base.models import \
    QueryResult, DocumentWithScore
from context_engine.context_engine.context_builder.stuffing import \
    StuffingContextBuilder


class TestStuffingContextBuilder:

    def setup_method(self):
        self.tokenizer = StubTokenizer()
        self.reference_metadata_field = "source"
        self.builder = StuffingContextBuilder(
            tokenizer=self.tokenizer,
            reference_metadata_field=self.reference_metadata_field)

        self.text1 = "I am a simple test string"
        self.text2 = "to check the happy path"
        self.text3 = "Another text string"
        self.text4 = "that needs to be tested"

        self.query_results = [
            QueryResult(query="test query 1",
                        documents=[
                            DocumentWithScore(id="doc1",
                                              text=self.text1,
                                              metadata={
                                                  "source": "test_source1"},
                                              score=1.0),
                            DocumentWithScore(id="doc2",
                                              text=self.text2,
                                              metadata={
                                                  "source": "test_source2"},
                                              score=1.0)
                        ]),
            QueryResult(query="test query 2",
                        documents=[
                            DocumentWithScore(id="doc3",
                                              text=self.text3,
                                              metadata={
                                                  "source": "test_source3"},
                                              score=1.0),
                            DocumentWithScore(id="doc4",
                                              text=self.text4,
                                              metadata={
                                                  "source": "test_source4"},
                                              score=1.0)
                        ])
        ]
        self.full_context = Context(content=[
            ContextQueryResult(query="test query 1",
                               snippets=[
                                   ContextSnippet(
                                       text=self.text1, reference="test_source1"),
                                   ContextSnippet(
                                       text=self.text2, reference="test_source2")
                               ]),
            ContextQueryResult(query="test query 2",
                               snippets=[
                                   ContextSnippet(
                                       text=self.text3, reference="test_source3"),
                                   ContextSnippet(
                                       text=self.text4, reference="test_source4")
                               ])
        ], num_tokens=0)
        self.full_context.num_tokens = self.tokenizer.token_count(
            self.full_context.to_text())

    def test_context_fits_within_max_tokens(self):
        context = self.builder.build(
            self.query_results, max_context_tokens=100)
        self.assert_num_tokens(context, 100)
        self.assert_contexts_equal(context, self.full_context)

    def test_context_exceeds_max_tokens(self):
        context = self.builder.build(self.query_results, max_context_tokens=30)

        expected_context = Context(content=[
            ContextQueryResult(query="test query 1",
                               snippets=[
                                   ContextSnippet(
                                       text=self.text1, reference="test_source1"),
                               ]),
            ContextQueryResult(query="test query 2",
                               snippets=[
                                   ContextSnippet(
                                       text=self.text3, reference="test_source3"),
                               ])
        ], num_tokens=0)
        expected_context.num_tokens = self.tokenizer.token_count(
            expected_context.to_text())

        self.assert_num_tokens(context, 30)
        self.assert_contexts_equal(context, expected_context)

    def test_context_exceeds_max_tokens_unordered(self):
        self.query_results[0].documents[0].text = self.text1 * 100
        context = self.builder.build(self.query_results, max_context_tokens=20)

        expected_context = Context(content=[
            ContextQueryResult(query="test query 2",
                               snippets=[
                                   ContextSnippet(
                                       text=self.text3, reference="test_source3"),
                               ])
        ], num_tokens=0)
        expected_context.num_tokens = self.tokenizer.token_count(
            expected_context.to_text())

        self.assert_num_tokens(context, 30)
        self.assert_contexts_equal(context, expected_context)

    def test_whole_query_results_not_fit(self):
        context = self.builder.build(self.query_results, max_context_tokens=10)
        assert context.num_tokens == 0
        assert context.content == []

    def test_max_tokens_zero(self):
        context = self.builder.build(self.query_results, max_context_tokens=0)
        self.assert_num_tokens(context, 0)
        assert context.content == []

    def test_empty_query_results(self):
        context = self.builder.build([], max_context_tokens=100)
        self.assert_num_tokens(context, 0)
        assert len(context.content) == 0

    def test_documents_with_duplicates(self):
        duplicate_query_results = self.query_results + [
            self.query_results[0]]

        # also duplicate doc1 within query 1
        duplicate_query_results[0].documents.append(
            duplicate_query_results[0].documents[0])

        context = self.builder.build(
            duplicate_query_results, max_context_tokens=100)
        self.assert_num_tokens(context, 100)
        self.assert_contexts_equal(context, self.full_context)

    def test_reference_metadata_missing(self):
        missing_metadata_query_results = [
            QueryResult(query="test missing metadata",
                        documents=[
                            DocumentWithScore(id="doc_missing_meta",
                                              text=self.text1,
                                              metadata={},
                                              score=1.0)
                        ])
        ]
        context = self.builder.build(
            missing_metadata_query_results, max_context_tokens=100)
        self.assert_num_tokens(context, 100)
        assert context.content[0].snippets[0].reference == ""

    def test_empty_documents(self):
        empty_query_results = [
            QueryResult(query="test empty doc",
                        documents=[
                            DocumentWithScore(id="empty_doc",
                                              text="",
                                              metadata={"source": "empty_source"},
                                              score=1.0)
                        ])
        ]
        context = self.builder.build(
            empty_query_results, max_context_tokens=100)
        self.assert_num_tokens(context, 0)
        assert context.content == []

    def assert_num_tokens(self, context: Context, max_tokens: int):
        assert context.num_tokens <= max_tokens
        assert self.tokenizer.token_count(
            context.to_text()) == context.num_tokens

    @staticmethod
    def assert_contexts_equal(actual: Context, expected: Context):
        assert actual.num_tokens == expected.num_tokens
        assert len(actual.content) == len(expected.content)
        for actual_qr, expected_qr in zip(actual.content, expected.content):
            assert actual_qr.query == expected_qr.query
            assert len(actual_qr.snippets) == len(expected_qr.snippets)
            for actual_snippet, expected_snippet in zip(actual_qr.snippets,
                                                        expected_qr.snippets):
                assert actual_snippet.text == expected_snippet.text
                assert actual_snippet.reference == expected_snippet.reference
