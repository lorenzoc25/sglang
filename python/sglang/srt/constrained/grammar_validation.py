"""Reject malformed constrained-decoding requests before HTTP streaming starts.

The OpenAI-compatible chat and text completion handlers and the
Anthropic-compatible Messages handler validate normalized constraints here before
tokenizer dispatch. Use the selected backend's native parser so syntax and schema
support match that backend, without compiling token masks.

Cache immutable specifications and success/error results, not request matchers.
The scheduler still owns tokenizer-specific compilation and remains the backstop
for failures that these syntax-only checks cannot detect.
"""

import json
from functools import lru_cache
from typing import Optional

from sglang.srt.constrained.base_grammar_backend import _grammar_key_contains_nul
from sglang.srt.constrained.json_schema_validation import (
    validate_outlines_json_schema,
    validate_xgrammar_json_schema,
)


class InvalidGrammarError(ValueError):
    """Carry the offending request field into the serving layer's HTTP 400."""

    def __init__(self, field: str, error: str):
        super().__init__(f"Invalid {field}: {error}")
        self.field = field
        self.error = error


@lru_cache(maxsize=128)
def validate_grammar(
    backend: str,
    key_type: str,
    key_string: str,
    *,
    any_whitespace: bool = True,
    whitespace_pattern: Optional[str] = None,
    max_whitespace_cnt: Optional[int] = None,
) -> Optional[str]:
    """Return an error string, or None for a valid spec or disabled backend.

    ``key_type`` uses scheduler grammar keys: json, regex, ebnf, structural_tag.
    ``key_string`` is serialized JSON for schemas/tags, or grammar source text.
    Whitespace settings must match scheduler compilation; they affect acceptance
    and are part of the LRU key, along with the backend and grammar specification.
    Both valid and invalid results are cached; callers attach request field names.
    """
    if backend == "none":
        return None

    try:
        # The scheduler uses the same guard before reaching native regex parsers.
        if _grammar_key_contains_nul(key_type, key_string):
            return "NUL bytes (\\u0000) are not allowed"

        if backend == "xgrammar":
            # Grammar.from_* parses syntax without building vocabulary masks.
            from xgrammar import Grammar

            if key_type == "json":
                # $$ANY$$ selects the scheduler's builtin JSON grammar, not a schema.
                if key_string != "$$ANY$$":
                    validate_xgrammar_json_schema(json.loads(key_string))
                    Grammar.from_json_schema(
                        key_string,
                        any_whitespace=any_whitespace,
                        max_whitespace_cnt=max_whitespace_cnt,
                    )
            elif key_type == "regex":
                Grammar.from_regex(key_string)
            elif key_type == "ebnf":
                Grammar.from_ebnf(key_string)
            elif key_type == "structural_tag":
                from sglang.srt.constrained.xgrammar_backend import (
                    XGrammarGrammarBackend,
                )

                Grammar.from_structural_tag(
                    XGrammarGrammarBackend.parse_structural_tag(key_string)
                )
        elif backend == "llguidance":
            # Validate serialized grammar without constructing a token matcher.
            from llguidance import LLMatcher, grammar_from

            if key_type == "json":
                serialized = LLMatcher.grammar_from_json_schema(
                    key_string,
                    defaults={
                        "whitespace_flexible": any_whitespace,
                        "whitespace_pattern": whitespace_pattern,
                    },
                )
            elif key_type == "structural_tag":
                from sglang.srt.constrained.llguidance_backend import GuidanceBackend

                serialized = GuidanceBackend.serialize_structural_tag(key_string)
            else:
                serialized = grammar_from(key_type, key_string)
            error = LLMatcher.validate_grammar(serialized)
            if error:
                return error
        elif backend == "outlines":
            import interegular
            from outlines_core.fsm.json_schema import build_regex_from_schema

            if key_type == "json":
                validate_outlines_json_schema(json.loads(key_string))
                regex = build_regex_from_schema(key_string, whitespace_pattern)
            elif key_type == "regex":
                regex = key_string
            else:
                return f"{key_type} is not supported by outlines"
            # Schema-to-regex conversion alone does not validate regex syntax.
            try:
                interegular.parse_pattern(regex)
            except (
                interegular.patterns.InvalidSyntax,
                interegular.patterns.Unsupported,
            ) as e:
                return str(e) or "Malformed regex specification"
    except (
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        IndexError,
        AssertionError,
    ) as e:
        # Cache error text; request-field attribution belongs to the serving layer.
        return str(e) or "Malformed grammar specification"
    return None
