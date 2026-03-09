from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class IngestTextBody(BaseModel):
    file_name: str = Field(examples=["Avatar: The Last Airbender"])
    text: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of the four elements—water, "
            "earth, fire or air—through practices known as 'bending', inspired by "
            "Chinese martial arts."
        ]
    )
    collection_name: str | None = Field(
        None,
        description="Optional collection name. If not provided, uses default collection.",
    )


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]


@ingest_router.post("/ingest", tags=["Ingestion"], deprecated=True)
def ingest(request: Request, file: UploadFile) -> IngestResponse:
    """Ingests and processes a file.

    Deprecated. Use ingest/file instead.
    """
    return ingest_file(request, file)


@ingest_router.post("/ingest/file", tags=["Ingestion"])
def ingest_file(
    request: Request,
    file: UploadFile,
    collection_name: str | None = None,
) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    Most common document
    formats are supported, but you may be prompted to install an extra dependency to
    manage a specific file type.

    A file can generate different Documents (for example a PDF generates one Document
    per page). All Documents IDs are returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). Those IDs
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    If `collection_name` is not provided, the collection will be auto-detected based on
    the file path (if available) or default collection will be used.
    """
    service = request.state.injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_bin_data(
        file.filename, file.file, collection_name=collection_name
    )
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.post("/ingest/text", tags=["Ingestion"])
def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    A Document will be generated with the given text. The Document
    ID is returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). That ID
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    If `collection_name` is not provided in the body, the default collection will be used.
    """
    service = request.state.injector.get(IngestService)
    if len(body.file_name) == 0:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_text(
        body.file_name, body.text, collection_name=body.collection_name
    )
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(
    request: Request, collection_name: str | None = None
) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.

    If `collection_name` is provided, only documents from that collection are returned.
    Otherwise, all documents are returned.
    """
    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested(collection_name=collection_name)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(
    request: Request, doc_id: str, collection_name: str | None = None
) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.

    If `collection_name` is provided, the document will be deleted from that collection.
    Otherwise, the default collection will be used.
    """
    service = request.state.injector.get(IngestService)
    service.delete(doc_id, collection_name=collection_name)


@ingest_router.get("/ingest/{doc_id}/file", tags=["Ingestion"])
def get_document_file(
    request: Request, doc_id: str, collection_name: str | None = None
) -> FileResponse:
    """Retrieve the original source file for a document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint or from
    the sources returned in chat/completion responses.

    This endpoint serves the original file that was ingested, allowing users to
    view the source document. For PDFs, you can append `#page=N` to the URL to
    open at a specific page.

    If `collection_name` is provided, the document will be retrieved from that collection.
    Otherwise, the default collection will be used.

    Returns:
        FileResponse: The original document file with appropriate Content-Type header

    Raises:
        HTTPException: 404 if document not found or file no longer exists
        HTTPException: 403 if file path is outside allowed directories
    """
    service = request.state.injector.get(IngestService)

    try:
        file_path = service.get_document_file_path(doc_id, collection_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # Determine Content-Type based on file extension
    content_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".ppt": "application/vnd.ms-powerpoint",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "application/xml",
        ".html": "text/html",
        ".htm": "text/html",
    }

    file_extension = file_path.suffix.lower()
    media_type = content_types.get(file_extension, "application/octet-stream")

    # Return file with inline disposition (view in browser) rather than download
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
        headers={"Content-Disposition": f'inline; filename="{file_path.name}"'},
    )
