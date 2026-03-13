import base64

from app.core.config import settings
from app.services.pdf_parser import parse_pdf
from app.services.text_chunker import build_chunk_records
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RagService:
    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

        self._vectorstore = Chroma(
            collection_name="rag_documents",
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_directory,
        )

        self._llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=0,
        )

    def ingest_pdf(self, file_path: str, source_name: str, namespace: str) -> dict:
        pages = parse_pdf(file_path=file_path, source_name=source_name)
        chunks = build_chunk_records(
            pages=pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(f"{namespace}:{source_name}:{chunk['chunk_id']}")
            texts.append(chunk["content"])
            metadatas.append(
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "namespace": namespace,
                }
            )

        if ids:
            self._vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            # Keep compatibility across Chroma client versions.
            if hasattr(self._vectorstore, "persist"):
                self._vectorstore.persist()

        return {
            "source": source_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "upserted": len(ids),
            "namespace": namespace,
        }

    def chat(self, question: str, namespace: str, top_k: int) -> dict:
        docs_and_scores = self._vectorstore.similarity_search_with_relevance_scores(
            question,
            k=top_k,
            filter={"namespace": namespace},
        )

        matches: list[dict] = []
        for doc, score in docs_and_scores:
            matches.append(
                {
                    "score": float(score) if score is not None else None,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                }
            )

        context = "\n\n".join(
            [
                f"[source={m['source']}, page={m['page']}, score={m['score']}]\n{m['content']}"
                for m in matches
            ]
        )

        prompt = (
            "You are a helpful assistant that answers only from provided context. "
            "If the context is insufficient, say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer in Indonesian with concise and clear wording."
        )
        response = self._llm.invoke(prompt)
        answer = response.content if isinstance(response.content, str) else str(response.content)

        return {
            "answer": answer,
            "sources": matches,
        }

    def _retrieve_matches(self, question: str, namespace: str, top_k: int) -> list[dict]:
        docs_and_scores = self._vectorstore.similarity_search_with_relevance_scores(
            question,
            k=top_k,
            filter={"namespace": namespace},
        )

        matches: list[dict] = []
        for doc, score in docs_and_scores:
            matches.append(
                {
                    "score": float(score) if score is not None else None,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                }
            )
        return matches

    def chat_with_pdf_attachment(
        self,
        question: str,
        pdf_path: str,
        source_name: str,
        namespace: str,
        top_k: int,
    ) -> dict:
        max_attachment_context_chars = 12000
        max_source_preview_chars = 800

        matches = self._retrieve_matches(question=question, namespace=namespace, top_k=top_k)
        attached_pages = parse_pdf(file_path=pdf_path, source_name=source_name)

        kb_context = "\n\n".join(
            [
                f"[kb source={m['source']}, page={m['page']}, score={m['score']}]\n{m['content']}"
                for m in matches
            ]
        )

        attachment_context = "\n\n".join(
            [f"[attachment page={p['page']}]\n{p['text']}" for p in attached_pages]
        )
        attachment_context = attachment_context[:max_attachment_context_chars]

        prompt = (
            "You are a helpful assistant that answers based on provided context. "
            "Use both knowledge-base context and attached PDF context. "
            "If the context is insufficient, say you do not know.\n\n"
            f"Knowledge-base context:\n{kb_context or '(empty)'}\n\n"
            f"Attached PDF context:\n{attachment_context or '(empty)'}\n\n"
            f"Question:\n{question}\n\n"
            "Answer in Indonesian with concise and clear wording."
        )

        response = self._llm.invoke(prompt)
        answer = response.content if isinstance(response.content, str) else str(response.content)

        attachment_sources = [
            {
                "score": None,
                "content": p["text"][:max_source_preview_chars],
                "source": source_name,
                "page": p["page"],
            }
            for p in attached_pages
        ]

        return {
            "answer": answer,
            "sources": matches + attachment_sources,
        }

    def chat_with_image_attachment(
        self,
        question: str,
        image_bytes: bytes,
        image_mime_type: str,
        source_name: str,
        namespace: str,
        top_k: int,
    ) -> dict:
        matches = self._retrieve_matches(question=question, namespace=namespace, top_k=top_k)

        kb_context = "\n\n".join(
            [
                f"[kb source={m['source']}, page={m['page']}, score={m['score']}]\n{m['content']}"
                for m in matches
            ]
        )

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Jawab dalam Bahasa Indonesia dengan ringkas dan jelas. "
                        "Gunakan konteks knowledge-base jika relevan. "
                        "Jika informasi tidak cukup, katakan tidak tahu.\n\n"
                        f"Knowledge-base context:\n{kb_context or '(empty)'}\n\n"
                        f"Question:\n{question}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                },
            ]
        )

        response = self._llm.invoke([message])
        answer = response.content if isinstance(response.content, str) else str(response.content)

        return {
            "answer": answer,
            "sources": matches
            + [
                {
                    "score": None,
                    "content": "Image attachment analyzed directly by vision model.",
                    "source": source_name,
                    "page": None,
                }
            ],
        }
