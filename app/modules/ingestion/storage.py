"""Local file storage service and PDF validation logic."""

import hashlib
import logging
from pathlib import Path
import re
from typing import Optional, Tuple
import uuid

from app.core.exceptions import FileValidationException, StorageException

logger = logging.getLogger(__name__)

PDF_MAGIC_BYTES = b"%PDF-"
EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class LocalStorageService:
    """Manages local filesystem storage, safe path resolution, and PDF file validation."""

    def __init__(self, upload_dir: str, max_file_size_bytes: int = 25 * 1024 * 1024) -> None:
        self.upload_dir = Path(upload_dir).resolve()
        self.max_file_size_bytes = max_file_size_bytes
        self._ensure_storage_directory()

    def _ensure_storage_directory(self) -> None:
        """Create upload directory if it does not exist."""
        try:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create upload directory %s: %s", self.upload_dir, str(exc))
            raise StorageException(f"Failed to create upload storage directory: {exc}") from exc

    def validate_pdf_content(self, content: bytes, original_filename: Optional[str] = None) -> Tuple[int, str]:
        """Validate PDF file size, signature, and integrity.

        Args:
            content: Raw byte content of the file
            original_filename: Original filename submitted by client

        Returns:
            Tuple of (file_size_bytes, sha256_hash)
        """
        # 1. Filename extension validation
        if original_filename:
            extension = Path(original_filename).suffix.lower()
            if extension != ".pdf":
                raise FileValidationException(
                    message=f"Only PDF files are supported. Received file with extension: '{extension}'",
                    error_code="INVALID_FILE_TYPE",
                )

        # 2. File size validation
        size_bytes = len(content)
        if size_bytes == 0:
            raise FileValidationException(
                message="Uploaded file is empty (0 bytes)",
                error_code="EMPTY_FILE",
            )
        if size_bytes > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            raise FileValidationException(
                message=f"File exceeds maximum allowed size of {max_mb:.1f} MB (received {actual_mb:.1f} MB)",
                error_code="FILE_TOO_LARGE",
            )

        # 3. PDF Magic Bytes / Signature Check
        if not content.startswith(PDF_MAGIC_BYTES):
            raise FileValidationException(
                message="Corrupted or invalid file: Missing standard '%PDF-' file signature",
                error_code="INVALID_PDF_SIGNATURE",
            )

        # 4. Basic Structural Integrity Check
        # A valid PDF must have minimum header and trailer structure (e.g. at least ~32 bytes and standard EOF or xref)
        if size_bytes < 32:
            raise FileValidationException(
                message="Corrupted or truncated PDF file (insufficient content)",
                error_code="CORRUPTED_PDF",
            )

        # Calculate SHA-256 hash for integrity and deduplication checks
        file_hash = hashlib.sha256(content).hexdigest()
        if file_hash == EMPTY_FILE_SHA256:
            raise FileValidationException(
                message="Uploaded file is empty (hash matches empty byte stream)",
                error_code="EMPTY_FILE",
            )
        return size_bytes, file_hash

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and unsafe characters."""
        # Strip all directory components
        basename = Path(filename).name
        # Replace non-alphanumeric characters (except dots, underscores, dashes)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", basename)
        # Ensure it has .pdf extension
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        return safe_name

    def save_file(self, document_id: str, original_filename: str, content: bytes) -> str:
        """Save file to local disk with unique ID and path traversal prevention.

        Args:
            document_id: Unique document UUID
            original_filename: Client filename
            content: File bytes

        Returns:
            Absolute stored file path as string
        """
        clean_basename = self.sanitize_filename(original_filename)
        stored_filename = f"{document_id}_{clean_basename}"
        target_path = (self.upload_dir / stored_filename).resolve()

        # Path traversal guard: verify target path is strictly within upload_dir
        if not target_path.is_relative_to(self.upload_dir):
            logger.error("Path traversal detected! Attempted path: %s", target_path)
            raise StorageException("Security error: Invalid file storage path detected")

        try:
            target_path.write_bytes(content)
            logger.info("Saved document %s to %s (%d bytes)", document_id, target_path, len(content))
            return str(target_path)
        except Exception as exc:
            logger.error("Failed to write file to disk %s: %s", target_path, str(exc))
            raise StorageException(f"Failed to write file to disk: {exc}") from exc

    def delete_file(self, file_path_str: str) -> bool:
        """Safely delete file from local storage, ensuring path stays inside upload_dir.

        Args:
            file_path_str: File path string to delete

        Returns:
            True if file existed and was deleted, False if not found
        """
        target_path = Path(file_path_str).resolve()

        # Security check: must reside inside upload_dir
        if not target_path.is_relative_to(self.upload_dir):
            logger.warning("Attempted deletion outside upload directory rejected: %s", target_path)
            return False

        if target_path.exists() and target_path.is_file():
            try:
                target_path.unlink()
                logger.info("Deleted stored file: %s", target_path)
                return True
            except Exception as exc:
                logger.error("Failed to delete stored file %s: %s", target_path, str(exc))
                raise StorageException(f"Failed to delete stored file: {exc}") from exc
        return False

    def stage_file_for_deletion(self, file_path_str: str) -> Optional[str]:
        """Stage a file for atomic deletion by renaming it to a temporary trash extension.

        Ensures that if downstream database deletion fails, the file can be cleanly restored.

        Args:
            file_path_str: Stored file path string

        Returns:
            Staged trash path string, or None if file does not exist on disk
        """
        target_path = Path(file_path_str).resolve()
        if not target_path.is_relative_to(self.upload_dir):
            logger.warning("Attempted staging outside upload directory rejected: %s", target_path)
            return None

        if not target_path.exists() or not target_path.is_file():
            return None

        trash_path = target_path.with_suffix(target_path.suffix + f".trash_{uuid.uuid4().hex[:8]}")
        try:
            target_path.rename(trash_path)
            logger.info("Staged file for deletion: %s -> %s", target_path, trash_path)
            return str(trash_path)
        except Exception as exc:
            logger.error("Failed to stage file for deletion %s: %s", target_path, str(exc))
            raise StorageException(f"Failed to stage file for deletion: {exc}") from exc

    def restore_staged_file(self, staged_path_str: str, original_path_str: str) -> bool:
        """Rollback: Restore a staged trash file back to its original path if database deletion fails.

        Args:
            staged_path_str: Temporary staged trash file path
            original_path_str: Original target file path

        Returns:
            True if restored, False otherwise
        """
        staged_path = Path(staged_path_str).resolve()
        original_path = Path(original_path_str).resolve()
        if staged_path.exists() and staged_path.is_file():
            try:
                staged_path.rename(original_path)
                logger.info("Successfully rolled back staged file: %s -> %s", staged_path, original_path)
                return True
            except Exception as exc:
                logger.critical(
                    "CRITICAL: Failed to rollback staged file %s to %s: %s",
                    staged_path,
                    original_path,
                    str(exc),
                )
                return False
        return False

    def finalize_staged_deletion(self, staged_path_str: str) -> bool:
        """Finalize physical file removal once database record has been safely deleted.

        Args:
            staged_path_str: Temporary staged trash file path

        Returns:
            True if removed, False otherwise
        """
        staged_path = Path(staged_path_str).resolve()
        if staged_path.exists() and staged_path.is_file():
            try:
                staged_path.unlink()
                logger.info("Finalized deletion of staged file: %s", staged_path)
                return True
            except Exception as exc:
                logger.warning("Could not unlink staged trash file %s: %s", staged_path, str(exc))
                return False
        return False
