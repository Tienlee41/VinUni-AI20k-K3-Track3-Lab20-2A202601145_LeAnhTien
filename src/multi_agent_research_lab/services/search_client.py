"""Search client with Tavily support and an offline corpus fallback."""

import json
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Search Tavily when configured, otherwise search bundled JSON knowledge."""

    def __init__(self, settings: Settings | None = None, corpus_root: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.corpus_root = corpus_root or self._find_corpus()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return deduplicated, ranked source documents."""

        if self.settings.tavily_api_key:
            try:
                results = self._tavily_search(query, max_results)
                if results:
                    return results
            except Exception:
                # Offline/local evidence is a safe fallback for transient search failures.
                pass
        return self._corpus_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        request = Request(
            "https://api.tavily.com/search",
            data=json.dumps(
                {
                    "api_key": self.settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            payload = response.read().decode("utf-8")
        response = json.loads(payload)
        return [
            SourceDocument(
                title=item.get("title", "Untitled source"),
                url=item.get("url"),
                snippet=item.get("content", item.get("snippet", "")),
                metadata={"source_id": item.get("url", "tavily"), "provider": "tavily"},
            )
            for item in response.get("results", [])
        ]

    def _corpus_search(self, query: str, max_results: int) -> list[SourceDocument]:
        if self.corpus_root is None or not self.corpus_root.exists():
            return [SourceDocument(title="Local fallback", snippet=f"No corpus found for: {query}")]
        terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        candidates: list[tuple[int, SourceDocument]] = []
        for path in sorted(self.corpus_root.glob("topics/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            topic = payload.get("topic", {})
            knowledge = payload.get("knowledge_base", {})
            entries: list[dict[str, Any]] = []
            entries.extend(knowledge.get("source_documents", []))
            entries.extend(knowledge.get("knowledge_articles", []))
            for entry in entries:
                text = " ".join(
                    str(entry.get(key, ""))
                    for key in (
                        "title",
                        "content",
                        "full_text",
                        "snippet",
                        "abstract",
                        "summary",
                        "key_takeaways",
                    )
                ).lower()
                score = sum(1 for term in terms if term in text)
                score += sum(1 for term in terms if term in str(topic.get("name", "")).lower())
                if score <= 0:
                    continue
                identifier = (
                    entry.get("document_id")
                    or entry.get("article_id")
                    or entry.get("citation_label")
                    or path.stem
                )
                snippet = entry.get("content") or entry.get("full_text") or entry.get("snippet")
                if not snippet and entry.get("key_takeaways"):
                    snippet = " ".join(str(item) for item in entry["key_takeaways"])
                candidates.append(
                    (
                        score,
                        SourceDocument(
                            title=str(entry.get("title", identifier)),
                            url=entry.get("url") or entry.get("provenance_url"),
                            snippet=str(snippet or entry.get("summary", ""))[:1200],
                            metadata={
                                "source_id": identifier,
                                "citation_label": entry.get("citation_label", identifier),
                                "article_id": entry.get("article_id"),
                                "is_synthetic": bool(entry.get("is_synthetic", False)),
                                "topic": topic.get("name", path.stem),
                                "provider": "offline_corpus",
                            },
                        ),
                    )
                )
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        result: list[SourceDocument] = []
        seen: set[str] = set()
        for _, document in candidates:
            identifier = str(document.metadata.get("source_id", document.title))
            if identifier in seen:
                continue
            seen.add(identifier)
            result.append(document)
            if len(result) >= max_results:
                break
        return result or [
            SourceDocument(title="Offline corpus", snippet=f"No matching evidence for: {query}")
        ]

    @staticmethod
    def _find_corpus() -> Path | None:
        for parent in [Path.cwd(), *Path.cwd().parents]:
            candidate = parent / "ai_agent_offline_research_corpus_v2"
            if candidate.exists():
                return candidate
        return None
