Overleaf build instructions for Handbook_V4_Presentation.tex

1. Upload the whole ZIP package to Overleaf, not only the .tex file.
   The project must contain:
   - Handbook_V4_Presentation.tex
   - figures/handbook_modes/*.png
   - figures/handbook_steps/*.png

2. In Overleaf, open Menu and set Compiler to XeLaTeX.
   Do not use pdfLaTeX for this Beamer/Metropolis template.

3. Set the main document to:
   Handbook_V4_Presentation.tex

4. If Overleaf reports "File ... not found", check that the figure folders
   were uploaded with the same paths used in the source:
   figures/handbook_modes/
   figures/handbook_steps/

5. If Overleaf reports a package/theme error, switch the project compiler to
   XeLaTeX first and recompile from scratch. The source includes fallbacks for
   optional packages, but Metropolis is available in a normal Overleaf TeX Live
   environment.

6. If the log mentions output.aux or \abx@aux@..., the project is reading a
   stale auxiliary file from an older BibLaTeX build. In Overleaf, use
   Recompile from scratch / Clear cached files, or upload this ZIP into a new
   blank project.
