"""Build the static article. Runtime has no dependencies or network requests."""
from pathlib import Path
import re, json, html, hashlib, shutil, csv
import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'serving'

def main():
    text = (ROOT/'article.md').read_text(encoding='utf-8')
    body = markdown.markdown(text, extensions=['tables','fenced_code','attr_list'])
    body = body.replace('<table>', '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable data table"><table>').replace('</table>', '</table></div>')
    body = re.sub(r'(<img src="([^"]+)"[^>]*>)', r'<div class="diagram-scroll" tabindex="0" role="region" aria-label="Diagram; scroll horizontally on a small screen"><a href="\2" aria-label="Open full-size diagram">\1</a></div>', body)
    heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    items = ''.join(f'<li><a href="#{i}">{t}</a></li>' for i,t in heads)
    words = len(re.findall(r'\b[\w’]+\b', re.sub('<[^>]+>', '', body)))
    minutes = round(words / 210)
    template = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow,noarchive"><meta name="referrer" content="strict-origin-when-cross-origin">
<meta name="description" content="A serving optimization worklog: expert scheduling, FP8 attention preparation, a Triton rounding bug, and what eight H200s actually measured.">
<meta name="theme-color" content="#faf9f5"><title>Learning to optimize a serving system · Pritish Saha</title>
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/style.css"><script src="assets/reader.js" defer></script></head>
<body id="top"><a class="skip" href="#article">Skip to article</a><div class="progress" aria-hidden="true"></div>
<header class="masthead"><a class="brand" href="https://pritishsaha92.github.io/">Pritish Saha</a><nav aria-label="Site"><a class="home-link" href="https://pritishsaha92.github.io/">About</a><a href="#reading">Reading notes</a><span class="draft">Systems notebook</span></nav></header>
<main><div class="hero"><div class="eyebrow">Systems notebook · Open Kernel Challenge</div>
<h1><span>Learning to optimize</span>a <em>serving system</em></h1>
<p class="subtitle">Expert scheduling, attention fusion, and the rounding bug<br class="desktop-break"> that changed how I think about GPU performance.</p>
<p class="byline">By Pritish Saha &nbsp;·&nbsp; 22 November 2026 &nbsp;·&nbsp; {minutes} min read</p><div class="hero-rule"></div></div>
<div class="opening"><div class="result-strip"><div><strong>1.0251×</strong><span>Best recorded serving score</span></div><div><strong>8 × H200</strong><span>Official evaluation node</span></div><div><strong>1 byte</strong><span>A mismatch worth investigating</span></div></div>
<details class="mobile-toc"><summary>In this article</summary><ol>{items}</ol></details></div>
<div class="layout"><nav class="toc" aria-label="Article contents"><div class="toc-label">In this article</div><ol>{items}</ol><a class="back" href="#top">Back to top</a></nav>
<article class="article" id="article">{body}</article></div></main>
<footer><span>Pritish Saha · Systems notebook, November 2026</span><a href="article.md" download>Download article source</a></footer></body></html>'''
    (ROOT/'index.html').write_text(template, encoding='utf-8')
    ledger = markdown.markdown((ROOT/'sources.md').read_text(encoding='utf-8'), extensions=['tables','attr_list'])
    ledger = re.sub(r'<h1>.*?</h1>', '', ledger, count=1)
    ledger = ledger.replace('<table>', '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable source table"><table>').replace('</table>', '</table></div>')
    ledger_heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', ledger)
    ledger_nav = ''.join(f'<a href="#{i}">{t}</a>' for i,t in ledger_heads)
    ledger_html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow,noarchive"><meta name="referrer" content="strict-origin-when-cross-origin">
<meta name="description" content="Annotated papers, pinned code, public reports and X discovery trail for the Muse serving worklog.">
<title>Sources &amp; research trail · Pritish Saha</title><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/style.css"></head>
<body id="top"><a class="skip" href="#source-ledger">Skip to source ledger</a>
<header class="masthead"><a class="brand" href="index.html">Pritish Saha</a><nav aria-label="Site"><a href="index.html#sources">Back to article</a><span class="draft">Systems notebook</span></nav></header>
<main class="ledger-main"><div class="hero ledger-hero"><div class="eyebrow">The notebook's bibliography</div><h1>Sources &amp; <em>research trail</em></h1><p class="subtitle">What informed the implementation.<br>What remained a lead. What to read next.</p></div>
<nav class="ledger-nav" aria-label="Source categories">{ledger_nav}</nav>
<article class="article ledger-article" id="source-ledger">{ledger}</article></main>
<footer><a href="index.html#sources">Return to the article</a><a href="sources.md" download>Download source ledger</a></footer></body></html>'''
    (ROOT/'sources.html').write_text(ledger_html, encoding='utf-8')
    print(json.dumps({'words':words,'reading_minutes':minutes,'sections':len(heads),'built':'index.html'}))

if __name__=='__main__': main()
