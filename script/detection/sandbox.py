"""inspect_event_log tool: evaluate model-written pandas code against the input
log in a restricted, read-only namespace."""
from __future__ import annotations

import ast
import collections
import itertools
import math
import re
import statistics

import pandas as pd

MAX_RESULT_CHARS = 4000

_SAFE_EVAL_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "range": range,
    "round": round, "abs": abs, "str": str, "int": int, "float": float,
    "bool": bool, "enumerate": enumerate, "zip": zip, "any": any, "all": all,
    "isinstance": isinstance, "type": type, "map": map, "filter": filter,
    "reversed": reversed, "frozenset": frozenset,
}

_FORBIDDEN_EXPR_PATTERNS = (
    "import", "__", "open(", "exec(", "eval(", "os.", "sys.",
    "subprocess", "globals(", "locals(", "compile(", "to_csv", "to_pickle",
)


def inspect_event_log(df: pd.DataFrame, expression: str) -> str:
    if not expression or not expression.strip():
        return "Error: empty expression."
    lowered = expression.lower()
    if any(bad in lowered for bad in _FORBIDDEN_EXPR_PATTERNS):
        return "Error: expression rejected (disallowed pattern). This tool is read-only."
    # `df`/`pd` must live in the globals dict, not locals: when eval()/
    # exec() is given separate globals/locals, a nested function defined
    # inside the code (e.g. `.apply(lambda s: ...)`) can only see names
    # from globals - locals-only names raise NameError inside the lambda
    # even though they're fine at the top level.
    namespace = {
        "__builtins__": _SAFE_EVAL_BUILTINS, "df": df, "pd": pd,
        # Pre-populated so common stdlib helpers (e.g. collections.defaultdict)
        # are usable without needing an `import` statement, which is
        # blocked outright by the forbidden-pattern filter above.
        "collections": collections, "itertools": itertools,
        "statistics": statistics, "math": math, "re": re,
    }
    try:
        # Accept ordinary multi-statement code (assignments, several
        # lines/semicolons), not just a single expression: parse it,
        # run every statement but the last with exec(), then evaluate
        # the last one as an expression to capture its value - this is
        # the same trick a REPL uses, and it's what lets the model write
        # code the normal way instead of one giant chained one-liner.
        tree = ast.parse(expression, mode="exec")
        if not tree.body:
            return "Error: empty expression."
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            if len(tree.body) > 1:
                exec(  # noqa: S102 - restricted namespace, read-only sandbox by design
                    compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<inspect_event_log>", "exec"),
                    namespace,
                )
            result = eval(  # noqa: S307 - restricted namespace, read-only sandbox by design
                compile(ast.Expression(last.value), "<inspect_event_log>", "eval"), namespace,
            )
            text = repr(result)
        else:
            exec(compile(tree, "<inspect_event_log>", "exec"), namespace)  # noqa: S102
            text = "(statements executed; end with an expression, e.g. a variable name, to see a value)"
    except Exception as e:
        text = f"Error evaluating expression: {e}"
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n... (truncated)"
    return text
