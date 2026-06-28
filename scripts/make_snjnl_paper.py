"""Assemble an Overleaf-ready Springer Nature (sn-jnl) version of the paper,
preserving the body verbatim from tri_hb_paper.tex. Compile on Overleaf with the
Springer Nature template (pdflatex + bibtex). Run: py -3.13 scripts/make_snjnl_paper.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "tri_hb_paper.tex"
OUT = ROOT / "paper" / "tri_hb_paper_snjnl.tex"

src = SRC.read_text(encoding="utf-8")

abstract = re.search(r"\\begin\{abstract\}\s*\\noindent\s*(.*?)\\end\{abstract\}", src, re.S).group(1).strip()
kw = re.search(r"\\textbf\{Keywords:\}\s*(.*?)\n\s*\n", src, re.S).group(1).strip()
keywords = re.sub(r"\s+", " ", kw).replace(";", ",").rstrip(".")

# Body: from the Introduction heading up to (not including) the Declarations.
body = src[src.index(r"\section{Introduction}"):src.index(r"\section*{Declarations}")].rstrip()
# Figures are referenced by basename for a flat Overleaf figures/ folder.
body = body.replace("../Handbook/figures/handbook_steps/", "").replace("../Handbook/figures/handbook_modes/", "")

PREAMBLE = r"""%% Springer Nature (sn-jnl) version for Rock Mechanics and Rock Engineering.
%% Compile on Overleaf with the "Springer Nature LaTeX Template" (sn-jnl.cls).
%% Build:  pdflatex -> bibtex -> pdflatex -> pdflatex
%% Upload alongside this file:  refs.bib  and a figures/ folder with all the
%% PNGs listed at the foot of this preamble.
%%
%% Reference style: sn-basic (Springer author-year, used by RMRE). To switch to
%% numbered references use [pdflatex,sn-mathphys-num] and \bibliographystyle is
%% then set by the class.
\documentclass[sn-basic]{sn-jnl}

\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{siunitx}
% siunitx v3 (Overleaf) removed \SI/\SIrange; provide them if absent.
\providecommand{\SI}[2]{\qty{#1}{#2}}
\providecommand{\SIrange}[3]{\qtyrange{#1}{#2}{#3}}
\providecommand{\diameter}{\ensuremath{\varnothing}}
\sisetup{detect-all}

\graphicspath{{figures/}}

\newcommand{\Cz}{C_{0}}
\newcommand{\eps}{\varepsilon}
\newcommand{\epsd}{\dot{\varepsilon}}

\begin{document}

\title[Monash True-Triaxial Hopkinson Bar and Computational Workspace]{Development of the
Monash True-Triaxial Hopkinson Bar System and an Integrated Computational Workspace for
Multiaxial Dynamic Rock Characterisation}

\author*[1]{\fnm{Qianbing} \sur{Zhang}}\email{qianbing.zhang@monash.edu}

\affil*[1]{\orgdiv{Department of Civil and Environmental Engineering},
\orgname{Monash University}, \orgaddress{\city{Clayton}, \state{VIC},
\postcode{3800}, \country{Australia}}}

\abstract{__ABSTRACT__}

\keywords{__KEYWORDS__}

\maketitle

"""

TAIL = r"""

\backmatter

\bmhead{Declarations}
\noindent\textbf{Funding.} Not declared (development study).\\[2pt]
\textbf{Competing interests.} The author declares no competing interests.\\[2pt]
\textbf{Data and code availability.} The Virtual Tri-HB workspace
(\texttt{tri\_hb\_integrated.py}) and the figure-generation scripts used here are
part of the Tri-HB repository. All figures in this paper are produced from
prescribed-pulse reference cases by the workspace's own analysis pipeline; no
experimental data were used.\\[2pt]
\textbf{Author contributions.} Q.B.Z. conceived the system and the workspace
framework, implemented the analysis pipeline, and wrote the manuscript.

\bibliography{refs}

\end{document}
"""

text = (PREAMBLE.replace("__ABSTRACT__", abstract).replace("__KEYWORDS__", keywords)
        + body + TAIL)
OUT.write_text(text, encoding="utf-8")

# List the figures the user must upload into figures/
figs = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body)))
print(f"Wrote {OUT}")
print(f"Body length: {len(body)} chars; {len(figs)} figures referenced:")
for f in figs:
    print("  figures/" + f if not f.endswith(".png") else "  figures/" + f)
