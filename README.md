# Code Are Added to master branch not main, just switch branch to master


Project is well beyond a beginner RAG implementation. It includes:

✅ Multi-PDF ingestion
✅ Token-based chunking
✅ ChromaDB vector storage
✅ Metadata (file and page)
✅ Semantic retrieval
✅ Streaming responses
✅ Conversation history
✅ Duplicate upload prevention
✅ Clean separation of ingestion, retrieval, and generation


**Aking questions about architecture:**

How should I organize retrieval?
Where should metadata live?
How should IDs be generated?
How should multiple PDFs be handled?
How should I structure context?
Should I chunk before metadata?
How should I assign IDs?
Should chunks belong to pages?
Should I store filenames?
How should retrieval work?


**Architecture Flow:**

files_reader()

↓

chunk_splitter()

↓

setup_db()

↓

retrieve_documents()

↓

build_prompt()

↓

send_message()
