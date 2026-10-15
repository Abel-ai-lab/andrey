"""The site's search and answer-engine metadata: structured data and sharing tags in each page's
head, and the root files crawlers read, ``robots.txt``, ``sitemap.xml``, and ``llms.txt``."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

# Built pages the sitemap leaves out: the search page and the symbol index carry no content of
# their own, and the drafts page lists unpublished posts.
UNLISTED = {"search", "genindex", "blog/drafts"}


def head_tags(
    page: dict,
    *,
    site_url: str,
    project: str,
    author: str,
    organization_url: str,
    organization_logo: str,
    software: dict,
    language: str = "en",
    post: dict | None = None,
    parents: Sequence[dict] = (),
) -> str:
    """The page's Twitter card fields, its robots rule (``noindex`` when ``page["indexable"]`` is
    false), a post's article tags, and one JSON-LD graph: the organization (its name, address, and
    logo's address), the website, the
    software, the page (a ``BlogPosting`` too for a post), and its breadcrumbs. A post or a release
    card describes the software at its own version (``post["version"]`` or
    ``page["software_version"]``); every other page, at ``software["version"]``. No element is
    visible."""
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
    # One release is one software node: the current release's keeps the plain id on every page.
    if written_for and written_for != software["version"]:
        software["version"] = written_for
        software_id = site_url + f"#software-{written_for}"
    else:
        software_id = site_url + "#software"
    graph = [
        {
            "@type": "Organization",
            "@id": organization_id,
            "name": author,
            "url": organization_url,
            "logo": organization_logo,
        },
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
    sections: Sequence[tuple[str, Sequence[str]]],
    *,
    site_url: str,
    project: str,
    version: str,
) -> str:
    """``llms.txt``: the project's name and summary, the current release, then each section's
    pages as links under its heading, each with the title and description its page carries
    (``entries``, from ``read_pages``). A page that is missing, private, or ``noindex`` raises
    ``ValueError``."""
    lines = [
        f"# {project}",
        "",
        "> " + " ".join(description.split()),
        "",
        f"Version: {version}.",
    ]
    listed = set()
    for heading, documents in sections:
        lines += ["", f"## {heading}", ""]
        for name in documents:
            if name in listed:
                continue
            listed.add(name)
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
    """A built page's ``meta`` tags by name or property, and whether it redirects."""

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.redirects = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag != "meta":
            return
        if (values.get("http-equiv") or "").lower() == "refresh":
            self.redirects = True
        name = values.get("property") or values.get("name")
        if name and values.get("content") is not None:
            self.meta[name] = values["content"]


def _head(target: Path) -> _PageHead:
    head = _PageHead()
    head.feed(target.read_text(encoding="utf-8").split("</head>", 1)[0])
    return head


def read_pages(outdir: Path, documents: Sequence[str]) -> dict[str, dict]:
    """Each document's title and description from its built page's preview tags, and whether it
    may be indexed. A page that is missing or has no preview tags raises ``ValueError``."""
    entries = {}
    for name in dict.fromkeys(documents):
        target = outdir / f"{name}.html"
        if not target.is_file():
            raise ValueError(f"Discovery document is missing: {name}")
        head = _head(target)
        if not head.meta.get("og:title") or not head.meta.get("og:description"):
            raise ValueError(f"Discovery document has no preview metadata: {name}")
        entries[name] = {
            "title": head.meta["og:title"],
            "description": head.meta["og:description"],
            "indexable": "noindex" not in head.meta.get("robots", "").lower(),
        }
    return entries


def sitemap_urls(outdir: Path) -> list[str]:
    """The address (``og:url``) of every built page a search engine should index, sorted: not a
    redirect, not ``noindex``, not in ``UNLISTED``, and outside the build's ``_`` folders."""
    urls = set()
    for target in outdir.rglob("*.html"):
        relative = target.relative_to(outdir)
        if any(part.startswith("_") or part == "PR" for part in relative.parts):
            continue
        if relative.with_suffix("").as_posix() in UNLISTED:
            continue
        head = _head(target)
        if head.redirects or "noindex" in head.meta.get("robots", "").lower():
            continue
        if url := head.meta.get("og:url"):
            urls.add(url)
    return sorted(urls)


def sitemap_xml(urls: Sequence[str]) -> str:
    """``sitemap.xml`` listing ``urls``."""
    entries = "".join(f"  <url><loc>{html.escape(url)}</loc></url>\n" for url in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}</urlset>\n"
    )


def robots_txt(site_url: str) -> str:
    """``robots.txt``: every crawler may read every page, and the sitemap's address."""
    return f"User-agent: *\nAllow: /\n\nSitemap: {site_url}sitemap.xml\n"
