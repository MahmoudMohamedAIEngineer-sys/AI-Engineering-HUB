"""Minimal dependency-free DOM reader for the two first-party roadmap sites.

The repository's roadmap pages are hand-written HTML with highly regular
`div class="..."` structures. We need their *data*, not their presentation, and we
must not modify the originals. `html.parser` from the standard library is enough;
this module only adds the small query surface the extractors need.

Text and child elements are kept in document order, because inline markup such as
`<strong>` inside a paragraph must not be allowed to reorder the sentence it belongs to.

Deliberately not a general HTML parser: no CSS selectors, no fragments, no reserialisation.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Dict, Iterator, List

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class Node:
    __slots__ = ("tag", "attrs", "content", "parent")

    def __init__(self, tag: str, attrs: Dict[str, str], parent: "Node | None"):
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.content: List[Any] = []  # str (text) | Node, in document order

    def add_text(self, text: str) -> None:
        self.content.append(text)

    def add_child(self, node: "Node") -> None:
        self.content.append(node)

    @property
    def children(self) -> List["Node"]:
        return [item for item in self.content if isinstance(item, Node)]

    @property
    def classes(self) -> List[str]:
        return self.attrs.get("class", "").split()

    def has_class(self, name: str) -> bool:
        return name in self.classes

    def attr(self, name: str, default: str = "") -> str:
        return self.attrs.get(name, default)

    def text(self) -> str:
        """Collapse all descendant text into single-spaced text (entity-decoded by the parser)."""
        return " ".join(self._raw_text().split())

    def own_text(self) -> str:
        """Text belonging to this element only, excluding text inside child elements."""
        return " ".join("".join(t for t in self.content if isinstance(t, str)).split())

    def _raw_text(self) -> str:
        parts: List[str] = []
        for item in self.content:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(item._raw_text())
        return "".join(parts)

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def find_all(self, class_name: str) -> List["Node"]:
        return [n for n in self.walk() if n.has_class(class_name)]

    def find(self, class_name: str) -> "Node | None":
        for node in self.walk():
            if node.has_class(class_name):
                return node
        return None

    def find_tag(self, tag: str) -> "Node | None":
        for node in self.walk():
            if node.tag == tag:
                return node
        return None


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", {}, None)
        self._stack: List[Node] = [self.root]

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        node = Node(tag, {k: (v or "") for k, v in attrs}, self._stack[-1])
        self._stack[-1].add_child(node)
        if tag not in VOID_ELEMENTS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: List[tuple]) -> None:
        self._stack[-1].add_child(Node(tag, {k: (v or "") for k, v in attrs}, self._stack[-1]))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].add_text(data)


def parse(html: str) -> Node:
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    return builder.root


def strip_scripts_and_styles(html_text: str) -> str:
    """Drop <script>/<style> blocks before parsing (their contents are not document data)."""
    import re

    html_text = re.sub(r"<script\b.*?</script>", "", html_text, flags=re.S | re.I)
    return re.sub(r"<style\b.*?</style>", "", html_text, flags=re.S | re.I)


def text_of(node: Node | None) -> str:
    return node.text() if node is not None else ""
