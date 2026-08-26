"""
WordPress content.rendered -> plain text.

A ~70-line html.parser subclass rather than BeautifulSoup. bs4 would be another
dependency (and pulls soupsieve) to do a job the stdlib does adequately here: the
input is WordPress's own generated markup, not arbitrary hostile HTML, so we do not
need bs4's error recovery.

The non-obvious part is which elements get DROPPED. WordPress renders figure captions,
embedded scripts and style blocks inline with the prose; keeping them means every
chunk carries "Foto: Dok. PYC" and the BM25 index learns that photo credits are
content.
"""

import html
import re
from html.parser import HTMLParser

# Content inside these is discarded entirely, tag and all.
_DROP = {"script", "style", "noscript", "figcaption", "iframe", "svg", "form"}
# These imply a line break when they open or close.
_BLOCK = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "section", "article", "header", "footer", "table", "ul", "ol",
}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _DROP:
            self._drop_depth += 1
        elif tag in _BLOCK and self._drop_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP:
            self._drop_depth = max(0, self._drop_depth - 1)
        elif tag in _BLOCK and self._drop_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(raw: str | None) -> str:
    """Strip markup and normalise whitespace, preserving paragraph boundaries."""
    if not raw:
        return ""
    parser = _Extractor()
    try:
        parser.feed(raw)
        parser.close()
        text = parser.text()
    except Exception:  # noqa: BLE001 -- malformed markup must degrade to a regex
                       # strip, never fail a whole harvest run for one bad post.
        text = re.sub(r"<[^>]+>", " ", raw)

    text = html.unescape(text)
    # Collapse runs of spaces/tabs but keep newlines, so the chunker can still see
    # paragraph boundaries -- that is what makes chunks split on meaning.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_title(title: str | None) -> str:
    """
    Lowercased, unaccented, punctuation-free title for trigram search and the
    title+year dedup fingerprint.
    """
    if not title:
        return ""
    text = html.unescape(title).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
