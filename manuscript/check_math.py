"""
Checks the mathematics of the repository's Markdown against what GitHub's renderer accepts.

    python manuscript/check_math.py                 # lint every tracked .md file (or the files given)
    python manuscript/check_math.py --fix           # rewrite the offending $...$ as $`...`$ where that repairs them
    python manuscript/check_math.py --attach-punctuation FILE   # "$x$," becomes "$x,$" (see Line breaks below)
    python manuscript/check_math.py --github        # after pushing: compare with GitHub's own rendering of main
    python manuscript/check_math.py --github --ref some-branch manuscript/open_classes.md

GitHub does not parse TeX before Markdown. It renders the Markdown first and then looks for `$...$` in the
resulting text, so a formula is lost whenever Markdown has touched it or its neighbourhood. The rules below were
measured against github.com on 2026-11-11 with three probe pages of about 600 constructions (see CLAUDE.md, which
states the same rules for writers). Every rule is an observed failure, not a style preference.

Rules for inline `$...$` (codes as printed):
  before    the opening $ must start the line or follow a space or "(" (or markup such as ** or a code span).
  after     the closing $ must not be followed by a letter or a digit.
  twin      the closing $ must not be followed by the character the formula ends with: "$f(x)$)" and "$b'$'s" fail.
  space     the formula must not begin or end with a space.
  italic    no formula inside *italics*, _italics_, <em> or <b> (bold with ** is fine), nor inside link text.
  escape    a backslash before ASCII punctuation loses the backslash: \\{ \\} \\\\ \\, \\; \\: \\! \\| \\% \\& \\# \\_
            become { } \\ , ; : ! | % & # _. Use \\lbrace \\rbrace \\thinspace \\enspace \\Vert \\newline, or $`...`$.
  under     an underscore after "}", ")", "]" or a space opens emphasis, and one before "{", "\\" or a space closes
            it; a pair of them in one paragraph, even in two formulas, destroys both. Keep a letter or a digit
            before every underscore, or use $`...`$.
  star      "*" pairs up as emphasis; write \\ast or \\star.
  tilde     "~" pairs up as strikethrough; write "\\ " or \\quad.
  angle     "<" can open an HTML tag that swallows the rest of the page; write \\lt (and \\gt). "&" only in ```math.
  pipe      "|" inside a table splits the cell; write \\mid or \\vert.
  link      "](" inside a formula becomes a link.
  dollars   an odd number of unescaped $ in a paragraph.
  anchor    a link `](#fragment)` or `](other.md#fragment)` whose fragment is the anchor of no heading.
Rules for display math:
  display   $$...$$ must be a paragraph of its own (blank lines around) and is subject to `escape`, `star`
            and `angle`; a ```math fence is verbatim and needs none of this.
  listmath  inside a list item a ```math fence is shown as code, and so is a $$...$$ that spans several lines:
            there, write the formula as $$...$$ on a single indented line with a blank line before and after.
The form $`...`$ is verbatim and works next to punctuation; it fails only glued to a letter or digit on its left
(`glued`) or as the last thing inside italics (`italic`).

Line breaks. GitHub lays each formula out as a box, and a browser may break the line on either side of a box, so
a comma or full stop written after the closing $ can start the next line. A word joiner does not prevent it
(tested in Chromium); writing the punctuation inside the formula does. `--attach-punctuation` moves a comma, full
stop or semicolon that directly follows a formula inside it (not a colon, which TeX would space as a relation).
It is a rewrite, not a rule: nothing is reported when the punctuation is outside.
"""
from __future__ import absolute_import
import argparse
import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
BACKTICK_MATH = re.compile(r'\$`([^`\n]+)`\$')
CODE_SPAN = re.compile(r'(`+)(?!`).+?(?<!`)\1(?!`)', re.S)
DISPLAY = re.compile(r'\$\$(.+?)\$\$', re.S)
LIST_ITEM = re.compile(r'^\s*([-*+]|\d+[.)])\s+')
CODE, MATH, DISPLAYED = '\ue000', '\ue001', '\ue002'      # masks for code spans, formulas and $$...$$ (never in a source)


def containers(lines):
    """(first line number, text, offset of the text in the file or None) for every inline container outside code
    fences: a paragraph, a heading, a list item, a table cell. Emphasis and the pairing of delimiters never cross
    these. The offset is None for a table row with a "|" inside a formula, which is reported as it stands."""
    out, current, start, fence = [], [], None, None
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line) + 1

    def flush():
        nonlocal current, start
        if current:
            out.append((start, '\n'.join(current), offsets[start - 1]))
        current, start = [], None

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        marker = re.match(r'^\s*(```+|~~~+)', line)
        if fence:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = None
            continue
        if marker:
            flush()
            fence = marker.group(1)
            if line[:1].isspace() and re.match(r'^\s*```+\s*math\s*$', line):
                out.append((number, line, -1))          # reported as `listmath` by the caller
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith('|'):
            flush()
            spans = [m for m in re.finditer(r'(?<!\\)\$(?!`)[^$\n]*?(?<!\\)\$', line) if '|' in m.group(0)]
            if spans:
                out.append((number, line, None))        # reported as `pipe` by the caller
            else:
                for m in re.finditer(r'[^|]+', line):
                    cell = m.group(0)
                    if cell.strip() and not set(cell.strip()) <= set('-: '):
                        lead = len(cell) - len(cell.lstrip())
                        out.append((number, cell.strip(), offsets[number - 1] + m.start() + lead))
            continue
        if stripped.startswith('#') or LIST_ITEM.match(line) or stripped.startswith('>'):
            flush()
        if start is None:
            start = number
        current.append(line)
    flush()
    return out


def flanking(text, i):
    """(can open, can close) for the delimiter character at text[i], by the CommonMark rules."""
    before = text[i - 1] if i > 0 else ' '
    after = text[i + 1] if i + 1 < len(text) else ' '
    left = not after.isspace() and (after not in PUNCT or before.isspace() or before in PUNCT)
    right = not before.isspace() and (before not in PUNCT or after.isspace() or after in PUNCT)
    if text[i] == '_':
        return left and (not right or before in PUNCT), right and (not left or after in PUNCT)
    return left, right


def italic_regions(masked):
    """Character ranges that Markdown renders in italics, on text whose formulas and code are masked."""
    regions = []
    for ch in '*_':
        opened = None
        for m in re.finditer(re.escape(ch) + '+', masked):
            if len(m.group(0)) % 2 == 0:
                continue                                  # ** is bold: formulas render inside it
            can_open, _ = flanking(masked, m.end() - 1)
            _, can_close = flanking(masked, m.start())
            if opened is None and can_open:
                opened = m.end()
            elif opened is not None and can_close:
                regions.append((opened, m.start()))
                opened = None
    for tag in ('em', 'b'):
        for m in re.finditer(rf'<{tag}>(.*?)</{tag}>', masked, re.S):
            regions.append((m.start(1), m.end(1)))
    return regions


def check_content(tex, where):
    problems = []
    escapes = sorted(set(re.findall(r'\\[!-/:-@\[-`{-~]', tex)))
    if escapes:
        problems.append(('escape', f"{' '.join(escapes)} lose the backslash {where}"))
    if '*' in tex:
        problems.append(('star', 'write \\ast or \\star instead of *'))
    if re.search(r'<|&', tex):
        problems.append(('angle', 'write \\lt for <; keep & for ```math blocks'))
    return problems


REPAIRABLE = {'before', 'twin', 'escape', 'under', 'italic', 'star', 'tilde', 'angle', 'link', 'space'}


def lint_text(text, repairs=None):
    """Problems of one Markdown source, as (line, rule, message, snippet). With `repairs` (a set), the file spans
    (start, end) of the `$...$` formulas that rewriting as $`...`$ repairs are added to it."""
    problems = []
    for start, body, base in containers(text.split('\n')):

        def report(offset, rule, message, snippet, span=None):
            problems.append((start + body.count('\n', 0, max(offset, 0)), rule, message, snippet.replace('\n', ' ')[:70]))
            if repairs is not None and span is not None and rule in REPAIRABLE and '`' not in body[span[0]:span[1]] \
                    and not (span[0] > 0 and body[span[0] - 1].isalnum()):
                repairs.add((base + span[0], base + span[1]))

        if base is None:
            report(0, 'pipe', 'a formula in a table row contains "|"; write \\mid or \\vert', body.strip())
            continue
        if base == -1:
            report(0, 'listmath', 'an indented ```math fence (inside a list item) is shown as code; write $$...$$ on one indented line', body.strip())
            continue
        masked = list(body)
        formulas = []                                     # (start, end, tex) of every `$...$` the author meant

        def mask(a, b, fill):
            for k in range(a, b):
                if masked[k] != '\n':
                    masked[k] = fill

        for m in BACKTICK_MATH.finditer(body):
            if m.start() > 0 and body[m.start() - 1].isalnum():
                report(m.start(), 'glued', '$`...`$ directly after a letter or digit is not rendered', m.group(0))
            formulas.append((m.start(), m.end(), None))
            mask(m.start(), m.end(), MATH)
        for m in CODE_SPAN.finditer(''.join(masked)):
            mask(m.start(), m.end(), CODE)
        for m in DISPLAY.finditer(''.join(masked)):
            if ''.join(masked).strip() != m.group(0).strip():
                report(m.start(), 'display', '$$...$$ needs a paragraph of its own (blank lines around)', m.group(0))
            elif body[:1].isspace() and '\n' in m.group(0):
                report(m.start(), 'listmath', 'inside a list item $$...$$ must be on a single line', m.group(0))
            for rule, message in check_content(m.group(1), 'in $$...$$; use a ```math fence'):
                report(m.start(), 'display' if rule == 'escape' else rule, message, m.group(0))
            mask(m.start(), m.end(), DISPLAYED)
        current = ''.join(masked)
        dollars = [m.start() for m in re.finditer(r'(?<!\\)\$', current)]
        if len(dollars) % 2:
            report(dollars[-1], 'dollars', 'odd number of $ in this paragraph (write \\$ for a currency sign)', current[max(0, dollars[-1] - 30):dollars[-1] + 30])
        for a, b in zip(dollars[0::2], dollars[1::2]):
            tex, shown = current[a + 1:b], current[a:b + 1]
            formulas.append((a, b + 1, tex))
            before = current[a - 1] if a > 0 else ' '
            after = current[b + 1] if b + 1 < len(current) else ' '
            if not (before.isspace() or before in ('(', '*', CODE, MATH) or (before == '>' and a >= 2)):
                report(a, 'before', f'opening $ after "{before}": only a space or "(" may precede it', current[max(0, a - 12):b + 1], (a, b + 1))
            if after.isalnum():
                report(b, 'after', f'closing $ followed by "{after}"', current[a:b + 8])
            elif tex and after == tex[-1] and not after.isspace():
                report(b, 'twin', f'the formula ends with "{after}" and is followed by "{after}"; add a space or use $`...`$', current[a:b + 4], (a, b + 1))
            if tex[:1].isspace() or tex[-1:].isspace() or not tex:
                report(a, 'space', 'the formula begins or ends with a space', shown, (a, b + 1))
            for rule, message in check_content(tex, 'in $...$; use the long macro or $`...`$'):
                report(a, rule, message, shown, (a, b + 1))
            if '~' in tex:
                report(a, 'tilde', 'write "\\ " or \\quad instead of ~', shown, (a, b + 1))
            if '](' in tex:
                report(a, 'link', '"](" inside a formula becomes a link', shown, (a, b + 1))
            mask(a + 1, b, MATH)
        # underscores: an opener followed by a closer, at least one of them in a `$...$` formula
        in_math = [False] * len(body)
        for a, b, tex in formulas:
            if tex is not None:
                for k in range(a, b):
                    in_math[k] = True
        raw = ''.join(c if m not in (CODE, DISPLAYED) and not (m == MATH and not in_math[k]) else 'x'
                      for k, (c, m) in enumerate(zip(body, masked)))
        opener = None
        for k, c in enumerate(raw):
            if c != '_':
                continue
            can_open, can_close = flanking(raw, k)
            if can_close and opener is not None and (in_math[k] or in_math[opener]):
                for position in (opener, k):
                    for a, b, tex in formulas:
                        if tex is not None and a <= position < b:
                            report(position, 'under', 'these two underscores pair up as emphasis; keep a letter or digit before "_" or use $`...`$',
                                   raw[max(0, opener - 14):opener + 6] + ' ... ' + raw[max(0, k - 8):k + 6], (a, b))
                opener = None
                continue
            if can_open and opener is None:
                opener = k
        final = ''.join(masked)
        for lo, hi in italic_regions(final):
            for a, b, tex in formulas:
                if lo <= a and b <= hi and (tex is not None or final[b:hi].strip() == ''):
                    report(a, 'italic', 'a formula inside italics is not rendered; move it out, use ** or $`...`$ followed by text', body[a:b],
                           (a, b) if tex is not None and final[b:hi].strip() else None)
        for m in re.finditer(r'\[[^\]\n]*\]\(', final):
            for a, b, tex in formulas:
                if tex is not None and m.start() < a and b <= m.end():
                    report(a, 'link', 'a formula inside link text is not rendered', body[a:b])
    return sorted(set(problems))


def heading_slugs(text):
    """The anchors GitHub gives the headings of a Markdown source: lower case, punctuation dropped, spaces to
    hyphens, and -1, -2, ... appended to a repeated heading."""
    slugs, seen, fence = set(), {}, False
    for line in text.split('\n'):
        if re.match(r'^\s*(```|~~~)', line):
            fence = not fence
        m = None if fence else re.match(r'^#{1,6} (.+)$', line)
        if not m:
            continue
        title = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', re.sub(r'`([^`]*)`', r'\1', m.group(1)))
        slug = re.sub(r'[^\w\- ]', '', title.strip().lower()).replace(' ', '-')
        if slug in seen:
            seen[slug] += 1
            slug = f'{slug}-{seen[slug]}'
        else:
            seen[slug] = 0
        slugs.add(slug)
    return slugs


def lint_anchors(text, path=None):
    """Links to a heading that does not exist, as (line, 'anchor', message, snippet): `](#fragment)` is looked up
    in the source itself and, when `path` is given, `](other.md#fragment)` in that other file."""
    problems, own, others, fence = [], heading_slugs(text), {}, False
    for number, line in enumerate(text.split('\n'), 1):
        if re.match(r'^\s*(```|~~~)', line):
            fence = not fence
        if fence:
            continue
        line = CODE_SPAN.sub(lambda m: ' ' * len(m.group(0)), line)      # a link written inside a code span is an example
        for m in re.finditer(r'\]\(([^)#\s]*)#([^)\s]+)\)', line):
            target, fragment = m.group(1), m.group(2)
            if not target:
                known = own
            elif path is not None and target.endswith('.md') and '://' not in target:
                other = (Path(path).parent / target.replace('%20', ' ')).resolve()
                if other not in others:
                    others[other] = heading_slugs(other.read_text(encoding='utf8')) if other.exists() else None
                known = others[other]
            else:
                continue
            if known is None or fragment not in known:
                problems.append((number, 'anchor', f'no heading has the anchor #{fragment}' + (f' in {target}' if target else ''),
                                 line[max(0, m.start() - 40):m.end()][:70]))
    return problems


# -- comparison with GitHub's own rendering ----------------------------------------------------------------------

def repository_slug():
    url = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    owner, name = re.sub(r'\.git$', '', url).rstrip('/').replace(':', '/').split('/')[-2:]
    return f'{owner}/{name}'


def github_rich_text(slug, ref, path):
    request = urllib.request.Request(f'https://github.com/{slug}/blob/{ref}/{path}', headers={'User-Agent': 'check-math'})
    page = urllib.request.urlopen(request, timeout=60).read().decode('utf8')
    m = re.search(r'<script type="application/json" data-target="react-app.embeddedData">(.*?)</script>', page, re.S)
    if not m:
        return None
    return json.loads(m.group(1)).get('payload', {}).get('codeViewBlobRoute', {}).get('richText')


def compare_with_github(slug, ref, path, source):
    """Formulas GitHub left as literal text, and formulas whose TeX it changed."""
    rich = github_rich_text(slug, ref, path)
    if rich is None:
        return None
    rendered = [html.unescape(html.unescape(s)) for s in re.findall(r'<math-renderer[^>]*>(.*?)</math-renderer>', rich, flags=re.S)]
    rest = re.sub(r'<math-renderer.*?</math-renderer>', '\u27e6M\u27e7', rich, flags=re.S)
    rest = re.sub(r'<pre.*?</pre>|<code.*?</code>|<section class="js-render-needs-enrichment.*?</section>', '', rest, flags=re.S)
    plain = html.unescape(re.sub(r'<[^>]+>', '', rest))
    missed = [m.group(0).strip() for line in plain.split('\n') for m in re.finditer(r'[^\u27e7]{0,40}\$[^\u27e6]{0,80}', line)]
    missed += ['```math shown as code: ' + html.unescape(m)[:80]
               for m in re.findall(r'<pre lang="math"[^>]*><code>(.*?)</code>', rich, flags=re.S)]
    anchors = set(re.findall(r'id="user-content-([^"]+)"', rich))
    missed += [f'link to a heading GitHub does not have: #{frag}' for frag in sorted(set(re.findall(r'href="#([^"]+)"', rich)) - anchors)]
    squeezed = re.sub(r'\s+', '', source)
    changed = [s for s in rendered if re.sub(r'\s+', '', s.strip('$')) not in squeezed]
    return len(rendered), missed, changed


def attach_punctuation(text):
    """Moves a comma, full stop or semicolon that directly follows a formula inside the formula, so that the
    browser cannot start a line with it. A colon stays outside: inside a formula TeX spaces it as a relation. Code
    fences, code spans and display formulas are left alone, and so is a formula that already ends with that
    character. Returns the new text and the number of formulas changed."""
    out, count, fence = [], 0, None
    for line in text.split('\n'):
        marker = re.match(r'^\s*(```+|~~~+)', line)
        if fence:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = None
            out.append(line)
            continue
        if marker:
            fence = marker.group(1)
            out.append(line)
            continue
        pieces = re.split(r'(\$`[^`\n]+`\$|`+[^`\n]+`+|\$\$[^\n]*?\$\$|(?<![\\$])\$(?![\s$`])[^$\n]*?(?<![\s\\])\$(?!\$))', line)
        for k in range(1, len(pieces), 2):
            piece, after = pieces[k], pieces[k + 1]
            verbatim = piece.startswith('$`')
            if not (verbatim or (piece.startswith('$') and not piece.startswith('$$'))):
                continue
            inner_end = -2 if verbatim else -1
            if after[:1] in tuple(',.;') and (len(after) == 1 or after[1] in ' \n)*"') and piece[inner_end - 1] != after[0] \
                    and not after.startswith('..'):
                pieces[k] = piece[:inner_end] + after[0] + piece[inner_end:]
                pieces[k + 1] = after[1:]
                count += 1
        out.append(''.join(pieces))
    return '\n'.join(out), count


def run_check_math():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help='Markdown files (default: every tracked .md file)')
    parser.add_argument('--github', action='store_true', help="compare with GitHub's rendering of the pushed files")
    parser.add_argument('--ref', default='main', help='branch or commit to compare with (default main)')
    parser.add_argument('--fix', action='store_true', help='rewrite the offending $...$ as $`...`$ where that repairs them')
    parser.add_argument('--attach-punctuation', action='store_true',
                        help='move a comma, full stop or semicolon that follows a formula inside it (files must be named)')
    args = parser.parse_args()
    if args.files:
        paths = [Path(f).resolve() for f in args.files]
    else:
        tracked = subprocess.run(['git', 'ls-files', '*.md'], cwd=ROOT, capture_output=True, text=True).stdout.split('\n')
        paths = [ROOT / f for f in tracked if f and not f.startswith('Older versions')]
    failures = 0
    for path in paths:
        source = path.read_text(encoding='utf8')
        name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        if args.attach_punctuation and args.files:
            source, moved = attach_punctuation(source)
            if moved:
                path.write_text(source, encoding='utf8')
            print(f'{name}: punctuation moved inside {moved} formulas')
        if args.fix:
            repairs = set()
            lint_text(source, repairs)
            for a, b in sorted(repairs, reverse=True):
                source = source[:a] + '$`' + source[a + 1:b - 1] + '`$' + source[b:]
            if repairs:
                path.write_text(source, encoding='utf8')
                print(f'{name}: {len(repairs)} formulas rewritten as $`...`$')
        problems = sorted(lint_text(source) + lint_anchors(source, path))
        for line, rule, message, snippet in problems:
            print(f'{name}:{line}: [{rule}] {message}\n      {snippet}')
        failures += len(problems)
        if args.github:
            result = compare_with_github(repository_slug(), args.ref, name.replace(' ', '%20'), source)
            if result is None:
                print(f'{name}: GitHub embeds no rendering for this file ({len(source.encode())} bytes: too large for '
                      f'the preview this check reads); the lint above is the only check')
                continue
            count, missed, changed = result
            print(f'{name}: GitHub rendered {count} formulas; {len(missed)} left as text, {len(changed)} with altered TeX')
            for snippet in missed:
                print(f'      not rendered: ...{snippet}')
            for tex in changed:
                print(f'      altered:      {tex[:100]}')
            failures += len(missed) + len(changed)
    print(f'{len(paths)} files, {failures} problems')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(run_check_math())
