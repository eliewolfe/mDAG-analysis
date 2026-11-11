# Instructions for writing in this repository

These instructions are permanent. They apply to every Markdown file here (the manuscript and notes in
`manuscript/`, `README.md`, `BRANCHES.md`), and they exist because the documents are read on github.com.

## Before committing any Markdown

```
python manuscript/refresh_links.py     # only after editing manuscript/piggybacks.md or renaming code it cites
python manuscript/check_math.py        # must print "0 problems"
```

After pushing, confirm against GitHub itself: `python manuscript/check_math.py --github` downloads GitHub's
rendering of each file on `main` and lists every formula it left as literal text or whose TeX it altered.
`manuscript/piggybacks.md` (234 kB) is too large for that preview, which GitHub leaves empty for large files, so
for it the lint is the check; to see it rendered, push it in parts to a scratch branch.

`python manuscript/check_math.py --fix` rewrites offending formulas in the verbatim form described below. Read the
diff afterwards.

## Mathematics on GitHub: what goes wrong and why

GitHub does not find formulas first. It renders the Markdown, and only then looks for `$...$` in the text that
results. So a formula is lost, or silently altered, whenever Markdown treats one of its characters, or a
neighbouring character, as markup. The rules below were measured on github.com on 2026-11-11 with about 600 probe
constructions; `manuscript/check_math.py` enforces all of them and its docstring lists them by code.

**Where a `$...$` formula may stand**

| Rule | Fails | Write instead |
|---|---|---|
| The opening `$` follows a space, a line start or `(`. Nothing else: not a quote, bracket, dash, slash or comma. | `"$x$"`, `[$x$]`, `$D$–$E$`, `C{0,$v$}` | ``"$`x`$"``, or reword |
| The closing `$` is not followed by a letter or digit. | `$n$th` | `$n$-th` |
| The closing `$` is not followed by the character the formula ends with. | `$f(x)$)`, `$b'$'s` | ``$`f(x)`$)`` |
| No formula inside italics (`*...*`, `_..._`, `<em>`, `<b>`) or inside link text. Bold `**...**` is fine. | `*with the latent $A$*` | `*with the latent* $A$` |

**What may stand inside `$...$`**

| Rule | Fails | Write instead |
|---|---|---|
| No backslash before punctuation: Markdown removes the backslash and the formula still renders, wrongly. | `\{ \}`, `\,` `\;` `\:` `\!`, `\\`, `\%` `\&` `\#` `\_`, and a backslash before a vertical bar | `\lbrace \rbrace`, `\thinspace` `\enspace` `\quad`, `\newline`, `\Vert`; or the verbatim form |
| A letter or digit before every `_`. After `}`, `)`, `]` or a space an underscore opens emphasis; two such in one paragraph, even in different formulas, destroy both. | `\mathrm{par}_{G}`, `(a)_{b}`, two of `\mathcal{C}_{G}` | `\mathcal C_G`, `F_S`, `x_{ij}`; or the verbatim form |
| No `*` (emphasis), no `~` (strikethrough). | `a*b*c`, `a~b~c` | `\ast`, `\star`; `\quad`, or a backslash and a space |
| No `<`: with a letter after it, it opens an HTML tag that swallows the rest of the page. | `$a<b>c$` | `\lt`, `\gt` |
| No vertical bar inside a table: it ends the cell. | a formula with a bar in a table cell | `\mid`, `\vert` |

**The verbatim form.** `` $`...`$ `` (a dollar, a backtick, the TeX, a backtick, a dollar) passes the TeX through
untouched and may stand next to any punctuation. Use it for any formula that cannot obey the rules above. Its two
limits: it must not be glued to a letter or digit on its left, and it must not be the last thing inside italics.

**Display mathematics.** Use a fenced block, which is verbatim:

````
```math
\lbrace a\rbrace_{b} \\ c
```
````

`$$...$$` is accepted only as a paragraph of its own (blank lines before and after) and is still subject to the
backslash rule, so `\\` and `\,` are lost in it.

Inside a list item the fence does not work: GitHub shows a fenced formula there as code. In a list, write the
formula as `$$...$$` on a single line, indented with the item, with a blank line before and after. A `$$...$$`
that spans several lines is not rendered there either.

**Punctuation after a formula.** GitHub lays each formula out as a box, and a browser may break the line on either
side of a box, so a comma written after the closing dollar can start the next line. Write a comma, full stop or
semicolon that follows a formula inside it: `$E\perp CD,$ the same`. Leave a colon outside (inside a formula TeX
spaces it as a relation). `python manuscript/check_math.py --attach-punctuation FILE` does this for a file.

**A literal dollar sign** is written `\$`.

## Words

Every proof that one structure inherits a quantum-classical gap from another has two halves. Name them in full and
say what they mean where they first appear; do not write "lift" or "pull-back".

* **The quantum direction**: a quantum model of the first structure can produce every quantum distribution of the
  structure with the known gap, in a form that has the extra observable properties the step relies on (for
  instance, one node perfectly predicts another).
* **The classical direction**: if a distribution with those properties were classical in the first structure, the
  corresponding distribution would be classical in the structure with the known gap.

Keep every explanation self-contained: define a term of the manuscript (seed, census, candidate, facet) in a clause
when a note for other readers uses it, or use plain words instead. The word "lift" remains only as the name of
the rule `QmDAG.lift` (a classical source replaced by a quantum one).

## Diagrams

Mermaid blocks render on GitHub. Keep each diagram small (under about ten nodes), quote any label that contains
punctuation (`FS["F_S"]`), and prefer several small diagrams to one large one.

## Tables of contents and links to sections

A note gets a table of contents by containing the two lines `<!-- contents -->` and `<!-- /contents -->`;
`python manuscript/refresh_links.py` fills the space between them with a nested list of the headings, each linked
to its anchor, and refreshes it whenever the headings change. `manuscript/check_math.py` reports every link
`](#anchor)` or `](other.md#anchor)` that matches no heading (rule `anchor`), so a renamed heading cannot leave a
dead link behind. Do not put a formula in a heading: its anchor is unpredictable.

## The manuscript's links

`manuscript/refresh_links.py` turns backticked names in `manuscript/piggybacks.md` into links to the code. It
indexes every function, class and attribute name in the repository root, `Special Applications/` and `tests/`, so a
new script there that defines a common word (`main`, `report`, `N`) hijacks that word wherever the manuscript has it
in backticks. After adding code, run the script and read the diff of the manuscript: only your intended lines may
change.

## Tests

`pytest` runs the fast suite; `pytest -m slow` runs the four-node census pins (several minutes). The two maintained
branches and what belongs to each are described in [BRANCHES.md](BRANCHES.md).
