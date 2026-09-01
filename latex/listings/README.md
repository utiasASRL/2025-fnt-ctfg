# Code listings

Each `<name>.tex` here is a `standalone` LaTeX document containing exactly one
C++ code listing.  Compiling it produces `<name>.pdf`, which `chap5.tex`
includes as a regular figure via the `\listingfig` helper defined in
`main.tex`:

```latex
\begin{figure}[ht!]
	\centering
	\listingfig{wnoa}
	\caption{Example setup for WNOA factors.}
	\label{fig:lst-wnoa}
\end{figure}
```

## Rebuilding

```sh
./build_listings.sh            # all listings
./build_listings.sh wnoa       # just one
```

or, from PowerShell:

```powershell
.\build_listings.ps1
.\build_listings.ps1 wnoa
```

Then recompile `main.tex`.  Intermediate files land in `build/`.

## Notes

* `preamble.tex` is shared by every listing.  It repeats the `C++Style`
  definition from `main.tex` and the font-size overrides from `nowfnt.cls`, so
  the rendered code matches the surrounding text at 1:1 scale.  The PDFs are
  therefore included at their natural size, *not* scaled with `width=`.

* Geometry.  A listing is wider than just its code: the line numbers hang to
  the left of the frame, and `frame=single` draws its rule `framesep +
  framerule` outside the code body on both sides.  Anything outside the
  standalone bounding box is silently cropped, so `preamble.tex` sizes the box
  to hold all of it — see the `\lstBodyWidth` / `\lstNumGutter` /
  `\lstFrameOver` block there.  `\lstBodyWidth` is `\textwidth` of the ROB
  journal layout, which keeps line wrapping identical to the old inline
  `lstlisting` output; if the page layout changes, that is the one number to
  update.

  Neither the standalone `varwidth` option nor a bare `minipage` is enough on
  its own: both end up measuring the width of the code body and leave the
  right-hand frame rule off the page.  The `listingbox` environment pins the
  box to its full width with an empty, zero-height `\hbox`.

  On the page, `\listingfig` (in `main.tex`) right-aligns the PDF with a
  `\lstfigoverhang` kern, which lands the code body exactly on the text block
  and lets the numbers hang into the left margin — the same position these
  listings had when they were inline.

* `firstnumber=` in `convert.tex`, `optimize.tex` and `cov_alt.tex` keeps the
  line numbering running across the sequence of figures.  These used to be
  `firstnumber=last`, which no longer works now that each listing is compiled
  on its own — if you add or remove lines, fix up the later `firstnumber`
  values by hand.

* A few code comments refer to other figures by number (e.g.
  `// Continues from Figure 5.3.`).  Those numbers are baked into the PDFs, so
  re-check them if the figure numbering in Chapter 5 changes.
