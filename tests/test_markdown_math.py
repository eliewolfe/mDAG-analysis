"""The Markdown of the repository must render on GitHub: manuscript/check_math.py lints it against the rules
measured on github.com (CLAUDE.md), and this test keeps every tracked document clean and pins the rules themselves."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_math", ROOT / "manuscript" / "check_math.py")
check_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_math)


def rules(text):
    return sorted({rule for _, rule, _, _ in check_math.lint_text(text)})


def test_every_tracked_document_is_clean():
    tracked = subprocess.run(['git', 'ls-files', '*.md'], cwd=ROOT, capture_output=True, text=True).stdout.split('\n')
    problems = {name: check_math.lint_text((ROOT / name).read_text(encoding='utf8'))
                for name in tracked if name and not name.startswith('Older versions')}
    assert {name: found for name, found in problems.items() if found} == {}


@pytest.mark.parametrize("text, expected", [
    ('aa $x+y$ bb, ($x$), **bold $x$**, $F_S$, $x_{ij}$ and $\\mathcal C_G$.', []),
    ('a quoted "$x$" formula', ['before']),
    ('a pair $D$–$E$ of nodes', ['before']),
    ('the $n$th node', ['after']),
    ('(in the form $F=(A,B)$) as', ['twin']),
    ("the copy $b'$'s parents", ['twin']),
    ('a formula $ x$ with a space', ['space']),
    ('*with the original latent $A$*', ['italic']),
    ('the set $\\{a,b\\}$ and the product $a\\,b$', ['escape']),
    ('x $\\mathrm{nondesc}_{G}(X)\\mid\\mathrm{par}_{G}(X)$ y', ['under']),
    ('x $\\mathcal{C}_{G}$ and $\\mathcal{I}_{G}$ y', ['under']),
    ('x $a*b*c$ y', ['star']),
    ('x $a~b~c$ y', ['tilde']),
    ('x $a<b>c$ y', ['angle']),
    ('| a | $p(a|b)$ |', ['pipe']),
    ('it costs $5 today', ['dollars']),
    ('text $$a_b$$ on the same line', ['display']),
    ('x $`\\{a\\}_{b}\\,c`$ y, "$`x`$" and *italic $`x`$ inside*', []),
    ('glued$`x`$ here', ['glued']),
    ('*italic ending in $`x`$*', ['italic']),
    ('```math\n\\{a\\}_{b} \\\\ c*d < e\n```', []),
])
def test_rules(text, expected):
    assert rules(text) == expected


def test_fix_rewrites_in_the_verbatim_form():
    text = 'the set $\\{a,b\\}$ in "$x$" and (see $f(y)$) here'
    repairs = set()
    check_math.lint_text(text, repairs)
    for a, b in sorted(repairs, reverse=True):
        text = text[:a] + '$`' + text[a + 1:b - 1] + '`$' + text[b:]
    assert text == 'the set $`\\{a,b\\}`$ in "$`x`$" and (see $`f(y)`$) here'
    assert check_math.lint_text(text) == []
