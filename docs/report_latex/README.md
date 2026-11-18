# IDSS project report: LaTeX version

LaTeX version of the final project report (`docs/report/final/IDSS_Project_Report.docx`), in the same college format
and with the same text, numbers, figures and tables. It is generated from the report content in `docs/report/*.md` by
`scripts/build_latex.py`. Do not edit the `.tex` files by hand: change the Markdown and run the script again.

| Path | Content |
|---|---|
| `main.tex` | Preamble (page layout, fonts, headings, captions, tables, code, footer, page index) and the `\input` list |
| `chapters/frontmatter.tex` | Title page, certificate, acknowledgement, page index, lists of figures and tables, abbreviations and nomenclature, abstract |
| `chapters/ch1_introduction.tex` … `ch8_conclusion.tex` | Chapters 1–8 |
| `chapters/references.tex`, `publications.tex` | References (`thebibliography`, same numbering as the Word report) and publications |
| `chapters/appendix_a.tex`, `appendix_b.tex` | Appendix A (user manual) and Appendix B (weekly progress) |
| `figures/` | The 23 images used by the report (diagrams, result charts, screenshots) |
| `IDSS_Report_Overleaf.zip` | `main.tex`, `chapters/` and `figures/`, ready to upload to Overleaf |

## Compile on Overleaf

1. Open [overleaf.com](https://www.overleaf.com) and choose **New Project → Upload Project**.
2. Upload `IDSS_Report_Overleaf.zip`.
3. Open **Menu** and set **Compiler: pdfLaTeX** and **Main document: main.tex**.
4. Click **Recompile**. Overleaf runs LaTeX as often as needed, so the page index and cross-references resolve
   on their own. Use **Download PDF** to save the result.

## Compile locally (MiKTeX or TeX Live)

```bash
cd docs/report_latex
latexmk -pdf main.tex
```

Without `latexmk`, run `pdflatex main.tex` three times. MiKTeX installs missing packages on first use.
`scripts/build_latex.py` compiles automatically when `latexmk` or `pdflatex` is on the PATH and saves the result as
`IDSS_Project_Report_LaTeX.pdf`.

## Regenerate from the Markdown

```bash
.venv\Scripts\python.exe scripts\build_latex.py
```

The script rewrites `main.tex`, `chapters/` and `figures/`, rebuilds the zip and then checks the output without TeX:
every file is ASCII with balanced braces and environments, every `\ref`, `\cite` and image resolves, the report has 23
figures, 40 tables, 8 algorithms and 5 listings, and each chapter has the same numbers and words as its Markdown
file. Use `--no-compile` to skip the PDF build.

## Format (same as the Word report)

- A4, `report` class, 12 pt, one-sided; margins top 1 in, bottom 1.15 in, left 1.5 in, right 0.6 in.
- Times (`newtxtext`, `newtxmath`), 1.5 line spacing, justified, 1 cm first-line indent (none after a heading).
- Chapter openings: "Chapter N" and the title, bold 16 pt, right-aligned, above a 2.25 pt rule, each on a new page.
  Sections 18 pt bold, subsections 14 pt bold.
- Captions in EB Garamond bold 12 pt, centred, numbered by chapter ("Figure 4.1:", "Table 4.1:"); figure captions
  below, table, algorithm and listing captions above.
- Tables with a grid and a bold, shaded header row. Short tables never split; long tables (`longtable`) repeat the
  header row on every page. Tables with 7–8 columns use 10 pt, 9 or more columns 9 pt.
- Code in Inconsolata 10 pt in a light grey box (`listings`); pseudo-code with `algorithm` + `algpseudocode`.
- Footer: college name and the centred page number. Title page unnumbered, certificate counted as page i without a
  number, roman numbers up to the nomenclature, Arabic numbers from the abstract.
- Page sequence: title page, certificate, acknowledgement, page index, list of figures, list of tables, abbreviations
  and nomenclature, abstract, chapters 1–8, references, publications and certificates, appendices A and B.

## College logo

The title page leaves an empty space for the logo. Put the image in `figures/` (for example `college_logo.png`), then
in `chapters/frontmatter.tex` uncomment the `\includegraphics` line and delete the `\vspace*{3cm}` line below it.
Re-running `build_latex.py` overwrites this change, so add the logo after the last build (or on Overleaf).

## Differences from the Word version

- Figures are floats (`[htbp]`): LaTeX may move a figure to the top of the next page within its section instead of
  leaving a gap, so page breaks differ slightly from the Word report. Short tables and algorithms stay in place.
- Pseudo-code uses `algpseudocode` layout: bold keywords and right-aligned "▷" comments instead of `//`.
- Fonts: Inconsolata instead of Consolas for code and EB Garamond instead of Garamond for captions (both are free
  fonts that pdfLaTeX can embed).
- Straight quotation marks are typeset as curly quotes. The rupee sign uses the `tfrupee` font; where it is not
  installed, "Rs." is printed instead.
