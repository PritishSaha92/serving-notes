"""Build the static article. Runtime has no dependencies or network requests."""
from pathlib import Path
import re, json, html, hashlib, shutil, csv
import markdown
from bs4 import BeautifulSoup
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'serving'

def main():
    text = (ROOT/'article.md').read_text(encoding='utf-8')
    body = markdown.markdown(text, extensions=['tables','fenced_code','attr_list','codehilite'], extension_configs={'codehilite':{'guess_lang':False,'css_class':'highlight'}})
    # Keep each note with its paragraph in the DOM. Grid places it in the margin
    # on wide screens; normal reading order puts it directly below on phones.
    soup = BeautifulSoup(body, 'html.parser')
    for number, note in enumerate(soup.select('aside.sidenote, div.reading'), 1):
        paragraph = note.find_previous_sibling()
        assert paragraph and paragraph.name == 'p', 'A margin note must follow its paragraph'
        row = soup.new_tag('div', attrs={'class':'annotation-row'})
        paragraphs = [paragraph]
        for _ in range(int(note.get('data-paragraphs', '1'))-1):
            previous = paragraphs[0].find_previous_sibling()
            assert previous and previous.name == 'p', 'Notes may span adjacent paragraphs only'
            paragraphs.insert(0, previous)
        following = []
        next_item = note
        for _ in range(int(note.get('data-follow', '0'))):
            next_item = next_item.find_next_sibling()
            assert next_item and next_item.name == 'p', 'A continued note must precede a paragraph'
            following.append(next_item)
        paragraphs[0].insert_before(row)
        main = soup.new_tag('div', attrs={'class':'note-prose'})
        for item in paragraphs: main.append(item.extract())
        row.append(main)
        marker = soup.new_tag('sup', attrs={'class':'note-marker','id':f'note-ref-{number}'})
        link = soup.new_tag('a', href=f'#note-{number}', attrs={'aria-label':f'Margin note {number}'})
        link.string = str(number); marker.append(link); paragraph.append(marker)
        for item in following: main.append(item.extract())
        note.extract(); note.name = 'aside'
        note['class'] = 'margin-note'; note['id'] = f'note-{number}'
        note['aria-label'] = f'Margin note {number}'
        back = soup.new_tag('a', href=f'#note-ref-{number}', attrs={'class':'note-number','aria-label':f'Back to passage {number}'})
        back.string = f'{number}.'; note.insert(0, back)
        row.append(note)
    body = str(soup)
    (ROOT/'assets/syntax.css').write_text(HtmlFormatter(style='friendly').get_style_defs('.highlight'),encoding='utf-8')
    body = body.replace('<table>', '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable data table"><table>').replace('</table>', '</table></div>')
    soup = BeautifulSoup(body, 'html.parser')
    for figure in soup.select('figure'):
        img = figure.find('img')
        if not img: continue
        wrap = soup.new_tag('div', attrs={'class':'diagram-scroll','tabindex':'0','role':'region','aria-label':'Diagram; scroll horizontally on a small screen'})
        img.wrap(wrap)
        anchor = soup.new_tag('a', href=img['src'], attrs={'aria-label':'Open full-size diagram'})
        img.wrap(anchor)
    body = str(soup)
    heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    items = ''.join(f'<li><a href="#{i}">{t}</a></li>' for i,t in heads)
    words = len(re.findall(r'\b[\w’]+\b', re.sub('<[^>]+>', '', body)))
    minutes = round(words / 210)
    template = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow,noarchive"><meta name="referrer" content="strict-origin-when-cross-origin">
<meta name="description" content="A serving optimization worklog: expert scheduling, FP8 attention preparation, a Triton rounding bug, and what eight H200s actually measured.">
<meta name="theme-color" content="#fffff8"><title>Learning to optimize a serving system · Pritish Saha</title>
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/style.css"><link rel="stylesheet" href="assets/syntax.css"><link rel="stylesheet" href="assets/interactives.css"><script src="assets/reader.js" defer></script><script src="assets/interactives.js" defer></script></head>
<body id="top"><a class="skip" href="#article">Skip to article</a><div class="progress" aria-hidden="true"></div>
<header class="masthead"><a class="brand" href="https://pritishsaha92.github.io/">Pritish Saha</a><nav aria-label="Site"><a class="home-link" href="https://pritishsaha92.github.io/">About</a><a href="#reading">Reading notes</a><span class="draft">Systems notebook</span></nav></header>
<main><div class="hero"><div class="eyebrow">Systems notebook · Open Kernel Challenge</div>
<h1>Learning to optimize<br>a <em>serving system</em></h1>
<p class="subtitle">Expert scheduling, attention fusion, and the rounding bug<br class="desktop-break"> that changed how I think about GPU performance.</p>
<p class="byline">Pritish Saha &nbsp;·&nbsp; 22 November 2026 &nbsp;·&nbsp; {minutes} min read<br>Interactive illustrations added 23 November 2026</p></div>
<div class="opening"><details class="contents"><summary>Contents <span>From the serving loop to a rounding boundary</span></summary><ol>{items}</ol></details></div>
<div class="layout">
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
    print(json.dumps({'words':words,'reading_minutes':minutes,'sections':len(heads),'margin_notes':len(soup.select('.margin-note')),'built':'index.html'}))

if __name__=='__main__': main()
