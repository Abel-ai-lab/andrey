"""Head metadata and discovery files derived from the site's existing page information."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath


def head_tags(
    page: dict,
    *,
    site_url: str,
    project: str,
    author: str,
    software: dict,
    language: str = "en",
    post: dict | None = None,
    parents: Sequence[dict] = (),
) -> str:
    """Supplement the existing preview tags without rendering any visible page content."""
    tags = {
        "twitter:title": page["title"],
        "twitter:description": page["description"],
        "twitter:image": page["image"],
    }
    if not page.get("indexable", True):
        tags["robots"] = "noindex, follow"
    if post:
        if post.get("date"):
            tags["article:published_time"] = post["date"]
        if post.get("authors"):
            tags["article:author"] = post["authors"][0]["url"]
    meta = "".join(
        f'<meta {"property" if key.startswith("article:") else "name"}="{key}" '
        f'content="{html.escape(value, quote=True)}">\n'
        for key, value in tags.items()
    )

    organization_id = site_url + "#organization"
    website_id = site_url + "#website"
    software = dict(software)
    written_for = (post or {}).get("version") or page.get("software_version")
    if written_for:
        software["version"] = written_for
    software_id = site_url + (f"#software-{written_for}" if written_for else "#software")
    graph = [
        {"@type": "Organization", "@id": organization_id, "name": author},
        {
            "@type": "WebSite",
            "@id": website_id,
            "url": site_url,
            "name": project,
            "inLanguage": language,
            "publisher": {"@id": organization_id},
        },
        {
            "@type": "SoftwareSourceCode",
            "@id": software_id,
            **software,
            "author": {"@id": organization_id},
            "url": site_url,
        },
    ]
    current = {
        "@type": "WebPage",
        "@id": page["url"] + "#page",
        "url": page["url"],
        "name": page["title"],
        "description": page["description"],
        "image": page["image"],
        "inLanguage": language,
        "isPartOf": {"@id": website_id},
        "about": {"@id": software_id},
    }
    if post:
        article = {
            **current,
            "@type": "BlogPosting",
            "@id": page["url"] + "#article",
            "headline": page["title"],
            "publisher": {"@id": organization_id},
            "mainEntityOfPage": {"@id": current["@id"]},
        }
        current["mainEntity"] = {"@id": article["@id"]}
        if post.get("date"):
            article["datePublished"] = post["date"]
        if post.get("authors"):
            article["author"] = [{"@type": "Person", **person} for person in post["authors"]]
        graph.append(article)
    crumbs = [{"title": project, "url": site_url}, *parents]
    if page["url"] != site_url:
        crumbs.append({"title": page["title"], "url": page["url"]})
    seen = set()
    items = []
    for crumb in crumbs:
        if crumb["url"] not in seen:
            seen.add(crumb["url"])
            items.append(
                {
                    "@type": "ListItem",
                    "position": len(items) + 1,
                    "name": crumb["title"],
                    "item": crumb["url"],
                }
            )
    breadcrumb_id = page["url"] + "#breadcrumbs"
    current["breadcrumb"] = {"@id": breadcrumb_id}
    graph.extend(
        [
            current,
            {"@type": "BreadcrumbList", "@id": breadcrumb_id, "itemListElement": items},
        ]
    )
    data = json.dumps({"@context": "https://schema.org", "@graph": graph}, ensure_ascii=False)
    # A title or summary must never terminate the HTML script containing its JSON string.
    for character, escaped in (
        ("<", r"\u003c"),
        (">", r"\u003e"),
        ("&", r"\u0026"),
        ("\u2028", r"\u2028"),
        ("\u2029", r"\u2029"),
    ):
        data = data.replace(character, escaped)
    return meta + f'<script type="application/ld+json">{data}</script>\n'


def llms_index(
    description: str,
    entries: dict[str, dict],
    documents: list[str],
    *,
    site_url: str,
    project: str,
    version: str,
) -> str:
    """List selected public pages using their existing titles, descriptions and HTML paths."""
    lines = [
        f"# {project}",
        "",
        "> " + " ".join(description.split()),
        "",
        f"Version: {version}.",
        "",
    ]
    for name in dict.fromkeys(documents):
        path = PurePosixPath(name)
        if (
            not name.startswith(("docs/", "blog/"))
            or ".." in path.parts
            or "PR" in path.parts
            or any(part.startswith("_") for part in path.parts)
        ):
            raise ValueError(f"Not a public document: {name}")
        if name not in entries:
            raise ValueError(f"Discovery document is missing: {name}")
        page = entries[name]
        if not page.get("indexable", True):
            raise ValueError(f"Discovery document is not indexable: {name}")
        title = " ".join(page["title"].split())
        title = re.sub(r"([\\\[\]])", r"\\\1", title)
        summary = " ".join(page["description"].split())
        lines.append(f"- [{title}]({site_url}{name}.html): {summary}")
    return "\n".join(lines) + "\n"


class _PageHead(HTMLParser):
    """Read the rendered preview metadata, including on an incremental Sphinx build."""

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta":
            name = values.get("property") or values.get("name")
            if name and values.get("content") is not None:
                self.meta[name] = values["content"]


def read_pages(outdir: Path, documents: Sequence[str]) -> dict[str, dict]:
    """Collect existing titles and descriptions from selected built HTML pages."""
    entries = {}
    for name in dict.fromkeys(documents):
        target = outdir / f"{name}.html"
        if not target.is_file():
            raise ValueError(f"Discovery document is missing: {name}")
        head = _PageHead()
        head.feed(target.read_text(encoding="utf-8").split("</head>", 1)[0])
        if not head.meta.get("og:title") or not head.meta.get("og:description"):
            raise ValueError(f"Discovery document has no preview metadata: {name}")
        entries[name] = {
            "title": head.meta["og:title"],
            "description": head.meta["og:description"],
            "indexable": "noindex" not in head.meta.get("robots", "").lower(),
        }
    return entries


def robots_txt() -> str:
    """The site's existing default public crawl policy, with no training-specific override."""
    return "User-agent: *\nAllow: /\n"
