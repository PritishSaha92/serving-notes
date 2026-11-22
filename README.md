# Learning to optimize a serving system

A standalone technical article by Pritish Saha about the Muse Spark Open Kernel Challenge serving project. The article is complete, with original editable diagrams, numbered margin notes, a tile-boundary explorer, source references and an evidence-backed results chart.

## Read it

Open **index.html** in a browser. It works directly from disk, including its local fonts, diagrams, links and interactive explorer. No build, network connection or JavaScript is required for the article text. External source links naturally require a connection.

For an HTTP preview:

```sh
python -m http.server 8080 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8080`. Keep this directory separate from `PritishSaha92.github.io` until you choose to integrate it.

## What is included

- `article.md`: canonical editable prose, with HTML figure embeds and explicit heading IDs.
- `index.html`: generated, fully readable static article.
- `sources.md`, `sources.html`: annotated bibliography and research trail, including primary papers, pinned source, relevant public reports, X discovery limits and the recorded reading scopes.
- `assets/style.css`, `assets/reader.js`: responsive design and optional enhancements.
- `assets/diagrams/`: eleven original SVG diagrams and matching **Excalidraw scenes**, plus a data-driven result chart. Open any `.excalidraw` file in Excalidraw to edit it. SVG exports are generated from the same source scene primitives; they are not screenshots from the reference blogs.
- `evidence/`: a small allowlist of project reports, chart data, an artifact hash inventory and claim map. Historical reports retain their original wording and dates.
- `tools/`: deterministic article/diagram builders, local development server and visual QA scripts.


## Edit and rebuild

```sh
python -m pip install -r requirements.txt
python tools/figures.py
python tools/build.py
```

The HTML is already built; dependencies are needed only to regenerate. `collect_evidence.py` needs the original sibling `serving/` tree and is not part of routine rebuilding. It fails on missing/empty evidence instead of silently inventing it.

To run browser QA:

```sh
npm ci
npx playwright install chromium
npm run qa
```

Set `CHROME_PATH` to an existing Chrome executable to use it instead. QA runs a temporary loopback server, checks desktop/mobile, anchors, image loading, overflow, the explorer, no-JavaScript reading and browser errors, and writes screenshots under `.preview/`. No browser is published or left running.


## Diagram and margin-note sources

`tools/sketches.py` draws the original hatched SVG illustrations and matching editable Excalidraw scenes. Labels embed the local OFL Kalam font; no diagram needs a remote font request. The results chart is generated separately from `evidence/results.csv`.

In `article.md`, a `sidenote` aside or `reading` block follows its associated paragraph. The build gives it a numbered anchor and aligns it in the right margin on wide screens. On smaller screens it appears directly below that paragraph, without requiring JavaScript. `data-paragraphs="2"` groups two adjacent paragraphs with a longer note; `data-follow="1"` keeps the next paragraph in the same reading block.

## Editorial decisions

This is a learning worklog, not a victory announcement. It reports best recorded score **1.0251** and final repaired score **1.0224**, credits the agents and public prior art, and distinguishes local L40 evidence from integrated H200 results. Course links are study companions; the source register does not support claiming every lecture was watched. The credited reference blogs influenced presentation, not copied prose or borrowed diagrams.

## License

Original article text, diagrams, site code and tooling are licensed under the [MIT License](LICENSE), copyright 2026 Pritish Saha. Bundled fonts retain their SIL Open Font Licenses, and the SGLang-derived evidence patch retains Apache 2.0. See [third-party notices and license scope](THIRD_PARTY_NOTICES.md).
