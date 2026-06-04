def chunk_documents(self):
        '''
    Uses MarkdownHeaderTextSplitter to chunk each document by header levels.
        For any chunk that has subsections in metadata, prepend higher-level headers
        to the front of page_content:
        - If meta has subsubsection (###), prepend "# section" and "## subsection"
        - If meta has subsection (##), prepend "# section"
        - If meta has only section (#), do nothing
        '''
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "section"),
                ("##", "subsection"),
                ("###", "subsubsection")
            ],
            strip_headers=False
        )

        for doc in self.documents:
            raw_chunks = splitter.split_text(doc.page_content)
            for c in raw_chunks:
                # keep parent doc metadata
                c.metadata.update(doc.metadata)

                # ---- Minimal addition: prepend higher-level headers if present ----
                meta = c.metadata
                prefix_lines = []

                # If this chunk is at ### level, prepend # and ##
                if meta.get("subsubsection"):
                    if meta.get("section"):
                        prefix_lines.append(f"# {meta['section']}")
                    if meta.get("subsection"):
                        prefix_lines.append(f"## {meta['subsection']}")

                # Else if this chunk is at ## level, prepend #
                elif meta.get("subsection"):
                    if meta.get("section"):
                        prefix_lines.append(f"# {meta['section']}")

                if prefix_lines:
                    # lstrip so we don't end up with extra leading blank lines
                    c.page_content = "\n\n".join(prefix_lines) + "\n\n" + c.page_content.lstrip()
                # ---- End minimal addition ----

            # Merge short chunks (unchanged)
            i = 0
            while i < len(raw_chunks):
                curr = raw_chunks[i]
                merged_content = curr.page_content.strip()
                merged_metadata = curr.metadata.copy()

                # Merge forward while content is too short
                while len(merged_content) < self.min_chars and i + 1 < len(raw_chunks):
                    next_chunk = raw_chunks[i + 1]
                    merged_content += "\n\n" + next_chunk.page_content.strip()
                    i += 1

                merged_doc = Document(page_content=merged_content, metadata=merged_metadata)
                self.chunks.append(merged_doc)
                i += 1

        # print(self.chunks[9])

    def chunk_by_top_header(self):
        '''
        Chunks each document by # (top-level) headers only.
        Returns a dict mapping filename -> list of chunk dicts.
        '''
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "section")],
            strip_headers=False
        )

        results = {}
        for doc in self.documents:
            filename = doc.metadata.get("source", "unknown")
            raw_chunks = splitter.split_text(doc.page_content)

            chunks = []
            for i, chunk in enumerate(raw_chunks):
                chunks.append({
                    "chunk_index": i,
                    "section": chunk.metadata.get("section", ""),
                    "content": chunk.page_content.strip(),
                    "metadata": doc.metadata,
                })
            results[filename] = chunks

        return results