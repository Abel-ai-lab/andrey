"""The site's search and answer-engine metadata (docs/discovery.py): head tags and structured data,
llms.txt, and sitemap.xml."""

import importlib.util
import json
import pathlib
from html.parser import HTMLParser

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("site_discovery", ROOT / "docs" / "discovery.py")
discovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(discovery)

SITE = "https://andrey.abel.ai/"
PAGE = {
    "title": "Release cards",
    "description": "Andrey's release cards, one per release.",
    "url": SITE + "docs/benchmarks.html",
    "image": SITE + "assets/preview/home.jpg",
}
SOFTWARE = {
    "name": "Andrey",
    "description": "Andrey, the fastest causal discovery package",
    "version": "0.4.0",
    "license": {
        "@type": "CreativeWork",
        "name": "Apache-2.0",
        "url": "https://github.com/Abel-ai-lab/andrey/blob/main/LICENSE",
    },
    "programmingLanguage": "Python",
    "runtimePlatform": "Python >=3.11",
    "codeRepository": "https://github.com/Abel-ai-lab/andrey",
    "sameAs": ["https://pypi.org/project/andrey-core/"],
}


class Head(HTMLParser):
    """The tags ``head_tags`` writes: ``meta`` by name or property, and the JSON-LD scripts. Any
    other element, or text outside a script, fails: the tags render nothing visible."""

    def __init__(self, text):
        super().__init__()
        self.meta = {}
        self.scripts = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            self.meta[attrs.get("name", attrs.get("property"))] = attrs.get("content")
        elif tag == "script":
            assert attrs.get("type") == "application/ld+json"
            self.scripts.append("")
        else:
            pytest.fail(f"The head tags include a visible element: {tag}")

    def handle_data(self, text):
        if self.scripts:
            self.scripts[-1] += text
        else:
            assert not text.strip()


def render(page=PAGE, **kwargs):
    return discovery.head_tags(
        page,
        site_url=SITE,
        project="Andrey",
        author="Abel AI Lab",
        organization_url="https://abel.ai/",
        organization_logo=SITE + "_static/abel-symbol.png",
        software=SOFTWARE,
        **kwargs,
    )


def graph(text):
    head = Head(text)
    assert len(head.scripts) == 1
    data = json.loads(head.scripts[0])
    assert data["@context"] == "https://schema.org"
    return {node["@type"]: node for node in data["@graph"]}


def test_the_twitter_card_repeats_the_page_preview():
    head = Head(render())
    assert head.meta["twitter:title"] == PAGE["title"]
    assert head.meta["twitter:description"] == PAGE["description"]
    assert head.meta["twitter:image"] == PAGE["image"]
    assert "description" not in head.meta  # conf.py's preview tags write the description
    assert "robots" not in head.meta


def test_the_graph_names_the_organization_the_software_and_the_page():
    nodes = graph(render())
    software = nodes["SoftwareSourceCode"]
    for key, value in SOFTWARE.items():
        assert software[key] == value
    assert nodes["Organization"] == {
        "@type": "Organization",
        "@id": SITE + "#organization",
        "name": "Abel AI Lab",
        "url": "https://abel.ai/",
        "logo": SITE + "_static/abel-symbol.png",
    }
    assert nodes["WebPage"]["name"] == PAGE["title"]
    assert nodes["WebPage"]["description"] == PAGE["description"]
    assert nodes["WebPage"]["about"]["@id"] == software["@id"] == SITE + "#software"
    assert "aggregateRating" not in software and "offers" not in software


def test_a_page_of_the_current_release_shares_its_software_node():
    # A post or release card of the current release is about the same software as every page.
    current = graph(render(post={"date": "2026-10-15", "authors": [], "version": "0.4.0"}))
    card = graph(render({**PAGE, "software_version": "0.4.0"}))
    for nodes in (current, card):
        assert nodes["SoftwareSourceCode"]["@id"] == SITE + "#software"
        assert nodes["SoftwareSourceCode"]["version"] == "0.4.0"


def test_breadcrumbs_follow_the_parent_pages():
    nodes = graph(render(parents=[{"title": "Andrey", "url": SITE + "docs/index.html"}]))
    items = nodes["BreadcrumbList"]["itemListElement"]
    assert [x["item"] for x in items] == [SITE, SITE + "docs/index.html", PAGE["url"]]
    assert [x["name"] for x in items] == ["Andrey", "Andrey", PAGE["title"]]
    assert [x["position"] for x in items] == [1, 2, 3]


def test_an_earlier_post_keeps_its_date_authors_and_release():
    post = {
        "date": "2026-10-08",
        "authors": [{"name": "Shu Wan", "url": SITE + "blog/author/shu-wan.html"}],
        "version": "0.1.0",
    }
    text = render(post=post)
    nodes = graph(text)
    article = nodes["BlogPosting"]
    assert article["headline"] == PAGE["title"]
    assert article["datePublished"] == post["date"]
    assert "dateModified" not in article
    assert article["author"] == [{"@type": "Person", **post["authors"][0]}]
    assert article["mainEntityOfPage"]["@id"] == nodes["WebPage"]["@id"]
    assert "breadcrumb" not in article
    assert nodes["SoftwareSourceCode"]["version"] == "0.1.0"
    assert nodes["SoftwareSourceCode"]["@id"] == SITE + "#software-0.1.0"
    head = Head(text)
    assert head.meta["article:published_time"] == post["date"]
    assert head.meta["article:author"] == post["authors"][0]["url"]


def test_a_title_cannot_close_its_script_or_add_an_element():
    title = '</script><img src=x onerror="alert(1)"> & <tag>'
    text = render({**PAGE, "title": title, "description": title})
    assert text.count("</script>") == 1
    assert "<img" not in text and "<tag>" not in text
    assert graph(text)["WebPage"]["name"] == title
    assert Head(text).meta["twitter:title"] == title


def test_a_page_that_is_not_indexable_is_marked_noindex():
    assert Head(render({**PAGE, "indexable": False})).meta["robots"] == "noindex, follow"


def llms(entries, sections):
    return discovery.llms_index(
        "The summary.", entries, sections, site_url=SITE, project="Andrey", version="0.4.0"
    )


def test_llms_txt_lists_each_section_under_its_heading():
    entries = {
        "docs/faq": {**PAGE, "title": "FAQ", "description": "The FAQ."},
        "docs/benchmarks/0.4.0": {**PAGE, "title": "0.4.0 card", "description": "The card."},
    }
    text = llms(entries, [("Docs", ["docs/faq"]), ("Release cards", ["docs/benchmarks/0.4.0"])])
    assert text == (
        "# Andrey\n\n> The summary.\n\nVersion: 0.4.0.\n\n"
        "## Docs\n\n- [FAQ](https://andrey.abel.ai/docs/faq.html): The FAQ.\n\n"
        "## Release cards\n\n"
        "- [0.4.0 card](https://andrey.abel.ai/docs/benchmarks/0.4.0.html): The card.\n"
    )


def test_llms_txt_refuses_a_missing_private_or_noindex_page():
    with pytest.raises(ValueError, match="missing"):
        llms({}, [("Docs", ["docs/missing"])])
    for path in ["search", "_modules/andrey/core", "docs/PR/private", "../private"]:
        with pytest.raises(ValueError, match="public"):
            llms({path: PAGE}, [("Docs", [path])])
    with pytest.raises(ValueError, match="indexable"):
        llms({"docs/faq": {**PAGE, "indexable": False}}, [("Docs", ["docs/faq"])])


def test_llms_txt_escapes_markdown_and_lists_a_page_once():
    entries = {"docs/faq": {**PAGE, "title": "FAQ [data]", "description": "One\nline"}}
    text = llms(entries, [("Docs", ["docs/faq", "docs/faq"]), ("More", ["docs/faq"])])
    assert text.count("https://andrey.abel.ai/docs/faq.html") == 1
    assert "[FAQ \\[data\\]]" in text and ": One line" in text


def page(path, head):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<html><head>{head}</head><body><p>Body.</p></body></html>")


def preview(url, title="T", description="D"):
    return (
        f'<meta property="og:title" content="{title}">'
        f'<meta property="og:description" content="{description}">'
        f'<meta property="og:url" content="{url}">'
    )


def test_read_pages_takes_the_preview_tags_of_the_built_page(tmp_path):
    page(tmp_path / "docs" / "faq.html", preview(SITE + "docs/faq.html", "FAQ &amp; data", "S."))
    entries = discovery.read_pages(tmp_path, ["docs/faq"])
    assert entries["docs/faq"] == {"title": "FAQ & data", "description": "S.", "indexable": True}
    page(tmp_path / "docs" / "faq.html", preview("x") + '<meta name="robots" content="noindex">')
    assert not discovery.read_pages(tmp_path, ["docs/faq"])["docs/faq"]["indexable"]


def test_read_pages_refuses_a_page_without_preview_tags(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        discovery.read_pages(tmp_path, ["docs/faq"])
    page(tmp_path / "docs" / "faq.html", "")
    with pytest.raises(ValueError, match="metadata"):
        discovery.read_pages(tmp_path, ["docs/faq"])


def test_the_sitemap_lists_the_indexable_pages_by_their_address(tmp_path):
    page(tmp_path / "index.html", preview(SITE))
    page(tmp_path / "docs" / "faq.html", preview(SITE + "docs/faq.html"))
    page(tmp_path / "search.html", preview(SITE + "search.html"))
    page(tmp_path / "genindex.html", preview(SITE + "genindex.html"))
    page(tmp_path / "blog" / "drafts.html", preview(SITE + "blog/drafts.html"))
    page(tmp_path / "_modules" / "a.html", preview(SITE + "_modules/a.html"))
    page(tmp_path / "docs" / "hidden.html", preview("h") + '<meta name="robots" content="noindex">')
    page(tmp_path / "old.html", '<meta http-equiv="refresh" content="0; url=docs/faq.html">')
    page(tmp_path / "moved.html", '<meta http-equiv="Refresh" content="0">' + preview("m"))
    urls = discovery.sitemap_urls(tmp_path)
    assert urls == [SITE, SITE + "docs/faq.html"]
    xml = discovery.sitemap_xml(urls)
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns=')
    assert f"<url><loc>{SITE}docs/faq.html</loc></url>" in xml
