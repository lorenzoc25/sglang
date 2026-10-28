"""CPU coverage for request-layer grammar validation with the native parsers."""

import json
import unittest

from sglang.srt.constrained.grammar_validation import validate_grammar
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SCHEMA = '{"type":"string"}'
_TAG = json.dumps(
    {
        "type": "structural_tag",
        "structures": [
            {"begin": "<call>", "schema": {"type": "string"}, "end": "</call>"}
        ],
        "triggers": ["<call"],
    }
)


class TestGrammarValidation(CustomTestCase):
    def setUp(self):
        validate_grammar.cache_clear()

    def test_invalid_grammars(self):
        for backend in ("xgrammar", "llguidance", "outlines"):
            for key_type, grammar in (
                ("regex", "["),
                ("json", '{"type":"invalid"}'),
                ("json", '{"type":'),
                ("ebnf", "root ::= missing"),
                ("structural_tag", '{"structures":[]}'),
                ("structural_tag", "[]"),
            ):
                with self.subTest(backend=backend, key_type=key_type, grammar=grammar):
                    error = validate_grammar(backend, key_type, grammar)
                    self.assertTrue(error)
                    self.assertNotIn("Traceback", error)

    def test_supported_valid_grammars(self):
        for backend in ("xgrammar", "llguidance", "outlines"):
            cases = [("regex", "(yes|no)"), ("json", _SCHEMA)]
            if backend != "outlines":
                cases.extend(
                    [("ebnf", 'root ::= "yes" | "no"'), ("structural_tag", _TAG)]
                )
            for key_type, grammar in cases:
                with self.subTest(backend=backend, key_type=key_type):
                    self.assertIsNone(validate_grammar(backend, key_type, grammar))

    def test_outlines_rejects_unsupported_grammar_kinds(self):
        for key_type, grammar in (("ebnf", 'root ::= "yes"'), ("structural_tag", _TAG)):
            with self.subTest(key_type=key_type):
                self.assertIn(
                    "not supported by outlines",
                    validate_grammar("outlines", key_type, grammar),
                )

    def test_backend_none_does_not_validate(self):
        for key_type in ("json", "regex", "ebnf", "structural_tag"):
            with self.subTest(key_type=key_type):
                self.assertIsNone(validate_grammar("none", key_type, "\x00["))

    def test_xgrammar_builtin_json_sentinel(self):
        self.assertIsNone(validate_grammar("xgrammar", "json", "$$ANY$$"))

    def test_nul_is_rejected_before_native_parsing(self):
        for backend in ("xgrammar", "llguidance", "outlines"):
            for key_type, grammar in (
                ("regex", "\x00"),
                ("json", '{"type":"string","pattern":"\\u0000"}'),
                ("ebnf", 'root ::= "\x00"'),
                ("structural_tag", _TAG.replace("<call>", "\\u0000")),
            ):
                with self.subTest(backend=backend, key_type=key_type):
                    self.assertIn("NUL", validate_grammar(backend, key_type, grammar))

    def test_backend_schema_prevalidation_is_preserved(self):
        for backend in ("xgrammar", "outlines"):
            with self.subTest(backend=backend):
                self.assertIn(
                    "multipleOf",
                    validate_grammar(
                        backend, "json", '{"type":"integer","multipleOf":2}'
                    ),
                )

    def test_xgrammar_legacy_null_schema_is_normalized(self):
        for schema in (None, {}):
            tag = json.loads(_TAG)
            tag["structures"][0]["schema"] = schema
            with self.subTest(schema=schema):
                self.assertIsNone(
                    validate_grammar("xgrammar", "structural_tag", json.dumps(tag))
                )

    def test_xgrammar_modern_structural_tag(self):
        for regex, valid in (("(yes|no)", True), ("[", False)):
            tag = json.dumps(
                {
                    "type": "structural_tag",
                    "format": {"type": "regex", "pattern": regex},
                }
            )
            with self.subTest(regex=regex):
                error = validate_grammar("xgrammar", "structural_tag", tag)
                self.assertEqual(error is None, valid)

    def test_structural_tag_trigger_must_match(self):
        tag = json.loads(_TAG)
        tag["triggers"] = ["unmatched"]
        for backend in ("xgrammar", "llguidance"):
            with self.subTest(backend=backend):
                self.assertTrue(
                    validate_grammar(backend, "structural_tag", json.dumps(tag))
                )

    def test_cache_reuses_valid_and_invalid_results(self):
        for grammar in ("[", "(yes|no)"):
            first = validate_grammar("xgrammar", "regex", grammar)
            self.assertEqual(first, validate_grammar("xgrammar", "regex", grammar))
        self.assertEqual(validate_grammar.cache_info().hits, 2)
        self.assertEqual(validate_grammar.cache_info().misses, 2)

    def test_cache_is_bounded(self):
        for i in range(200):
            self.assertIsNone(validate_grammar("xgrammar", "regex", f"yes_{i}"))
        self.assertEqual(validate_grammar.cache_info().currsize, 128)

    def test_cache_separates_backend_and_whitespace_options(self):
        for backend in ("xgrammar", "llguidance", "outlines"):
            for any_whitespace in (False, True):
                self.assertIsNone(
                    validate_grammar(
                        backend, "json", _SCHEMA, any_whitespace=any_whitespace
                    )
                )
        self.assertEqual(validate_grammar.cache_info().misses, 6)


if __name__ == "__main__":
    unittest.main()
