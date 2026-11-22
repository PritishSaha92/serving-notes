# License scope and third-party notices

Copyright (c) 2026 Pritish Saha.

The root MIT license applies to the author's original article text, diagrams,
site code, build tools and documentation. It does not relicense third-party
material, upstream-derived code, or material merely linked from the article.

## Fonts

The bundled Lora, DM Sans and Kalam fonts remain under the SIL Open Font License 1.1.
Their copyright and license notices are retained in:

- `assets/fonts/lora-OFL.txt`
- `assets/fonts/dmsans-OFL.txt`
- `assets/fonts/kalam-OFL.txt`

Kalam is embedded in the SVG teaching diagrams so their labels render offline.
Font source: https://github.com/google/fonts/tree/main/ofl/kalam

## SGLang-derived patch

`evidence/repair-delta.patch` and `evidence/mapping.patch` are historical
deltas for the SGLang-derived implementation. They and their upstream code
context remain under Apache License
2.0, not the root MIT license. The license text is retained in
`licenses/SGLang-Apache-2.0.txt`.

Upstream: https://github.com/sgl-project/sglang

Base revision: `297e54fd105e974cdf401e5b4d20253a22c16f90`.

The repair delta selects stock-compatible launch tiles and retains the
stock fallback for an unsupported compilation family. The mapping delta adds
the initial Triton global-to-local expert-ID lookup. The project used coding
and research agents for implementation and validation. The complete code and
its original notices belong in the separate implementation repository; this
notebook includes only the historical patch evidence.

## References and dependencies

External papers, articles, repositories and posts retain their respective
authors' rights and licenses. Attribution appears in `sources.md` and the
article. The original diagrams here are not reproductions of those sources.
Archived project reviews do not grant additional rights over upstream code
or third-party material they discuss or excerpt.

Development dependencies retain their own licenses. They are not bundled as
newly MIT-licensed work by this repository.
