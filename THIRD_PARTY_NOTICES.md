# License scope and third-party notices

Copyright (c) 2026 Pritish Saha.

The root MIT license applies to the author's original article text, diagrams,
site code, build tools and documentation. It does not relicense third-party
material, upstream-derived code, or material merely linked from the article.

## Fonts

The bundled Lora and DM Sans fonts remain under the SIL Open Font License 1.1.
Their copyright and license notices are retained in:

- `assets/fonts/lora-OFL.txt`
- `assets/fonts/dmsans-OFL.txt`

## SGLang-derived patch

`evidence/repair-delta.patch` is a historical delta for the SGLang-derived
implementation. It and its upstream code context remain under Apache License
2.0, not the root MIT license. The license text is retained in
`licenses/SGLang-Apache-2.0.txt`.

Upstream: https://github.com/sgl-project/sglang

Base revision: `297e54fd105e974cdf401e5b4d20253a22c16f90`.

The changes in the delta select stock-compatible launch tiles and retain the
stock fallback for an unsupported compilation family. The project used coding
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
