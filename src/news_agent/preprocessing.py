import hashlib
import re
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li", "h1", "h2"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    parser = _TextExtractor()
    parser.feed(text)
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def content_hash(text):
    return hashlib.sha256(clean_text(text).casefold().encode("utf-8")).hexdigest()


def prepare_documents(records):
    seen, ids, output = set(), {}, []
    for record in records:
        body = clean_text(record.get("text", ""))
        if not body:
            continue
        digest = content_hash(body)
        doc_id = str(record.get("document_id") or digest[:20])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", doc_id):
            raise ValueError("document_id must contain only letters, numbers, _, . or -")
        if doc_id in ids and ids[doc_id] != digest:
            raise ValueError(f"Same document ID has different content: {doc_id}")
        ids[doc_id] = digest
        if digest in seen:
            continue
        seen.add(digest)
        output.append({"document_id": doc_id, "text": body, "content_sha256": digest,
                       "title": clean_text(record.get("title", "")),
                       "source": str(record.get("source") or ""),
                       "published": str(record.get("published") or "")})
    return output


def chunk_documents(documents, chunk_size=160, overlap=30):
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    chunks = []
    for document in documents:
        words = document["text"].split()
        for number, start in enumerate(range(0, len(words), chunk_size - overlap)):
            chunks.append({**document, "chunk_id": f"{document['document_id']}:{number}",
                           "text": " ".join(words[start:start + chunk_size])})
            if start + chunk_size >= len(words):
                break
    return chunks
