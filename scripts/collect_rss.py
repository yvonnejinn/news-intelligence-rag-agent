import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import pandas as pd
from news_agent.preprocessing import clean_text, content_hash, prepare_documents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feed", action="append", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import feedparser
    rows = []
    for feed_url in args.feed:
        if urlparse(feed_url).scheme not in {"https", "http"}:
            raise ValueError("RSS feed must be an HTTP(S) URL")
        request = Request(feed_url, headers={"User-Agent": "NewsPortfolioResearch/1.0"})
        with urlopen(request, timeout=30) as response:
            data = response.read(5_000_001)
        if len(data) > 5_000_000:
            raise ValueError("RSS feed exceeds 5 MB")
        feed = feedparser.parse(data)
        if not feed.entries:
            raise ValueError(f"No RSS entries parsed from {feed_url}")
        for entry in feed.entries:
            title, body = clean_text(entry.get("title", "")), clean_text(entry.get("summary", ""))
            text = f"{title}. {body}" if body else title
            if text:
                rows.append({"document_id": content_hash(text)[:20], "title": title, "text": text,
                             "source": entry.get("link", feed_url),
                             "published": entry.get("published", entry.get("updated", ""))})
    output = pd.DataFrame(prepare_documents(rows))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Wrote {len(output)} unique summaries to {path}")


if __name__ == "__main__":
    main()
