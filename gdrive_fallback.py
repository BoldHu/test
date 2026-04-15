from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


@dataclass
class UploadResult:
    action: str
    file_id: str
    local_path: str
    remote_name: str


class GoogleDriveUploader:
    def __init__(self, service_account_json: str, folder_id: str):
        self.service_account_json = service_account_json
        self.folder_id = folder_id
        self._service = None

    def enabled(self) -> bool:
        return bool(self.service_account_json and self.folder_id)

    def _build_service(self):
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials.from_service_account_file(
            self.service_account_json,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    @property
    def service(self):
        if self._service is None:
            if not os.path.isfile(self.service_account_json):
                raise FileNotFoundError(
                    f"Google Drive service account file not found: {self.service_account_json}"
                )
            self._service = self._build_service()
        return self._service

    def _find_existing_file_id(self, remote_name: str) -> str | None:
        escaped_name = remote_name.replace("'", "\\'")
        query = (
            f"'{self.folder_id}' in parents and trashed = false and "
            f"name = '{escaped_name}'"
        )
        response = (
            self.service.files()
            .list(
                q=query,
                fields="files(id, name)",
                pageSize=1,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        if not files:
            return None
        return files[0]["id"]

    def upload_file(self, local_path: str, remote_name: str | None = None) -> UploadResult:
        from googleapiclient.http import MediaFileUpload

        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        remote_name = remote_name or os.path.basename(local_path)
        media = MediaFileUpload(local_path, resumable=True)
        existing_id = self._find_existing_file_id(remote_name)

        if existing_id is not None:
            response = (
                self.service.files()
                .update(
                    fileId=existing_id,
                    media_body=media,
                    supportsAllDrives=True,
                )
                .execute()
            )
            return UploadResult(
                action="updated",
                file_id=response["id"],
                local_path=local_path,
                remote_name=remote_name,
            )

        response = (
            self.service.files()
            .create(
                body={"name": remote_name, "parents": [self.folder_id]},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        return UploadResult(
            action="created",
            file_id=response["id"],
            local_path=local_path,
            remote_name=remote_name,
        )


def init_gdrive_uploader(
    *,
    enabled: bool,
    service_account_json: str,
    folder_id: str,
):
    if not enabled:
        return None

    try:
        uploader = GoogleDriveUploader(
            service_account_json=service_account_json,
            folder_id=folder_id,
        )
        if not uploader.enabled():
            print("[warn] Google Drive fallback requested, but credentials or folder id are missing. Skipping upload.")
            return None
        _ = uploader.service
        print(f"Google Drive fallback enabled for folder id: {folder_id}")
        return uploader
    except Exception as exc:
        print(f"[warn] Failed to initialize Google Drive fallback: {exc}")
        return None


def upload_artifacts_to_gdrive(uploader, paths: Iterable[str]) -> None:
    if uploader is None:
        return

    seen = set()
    for path in paths:
        if not path or path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            result = uploader.upload_file(path)
            print(
                f"[gdrive] {result.action} {result.remote_name} "
                f"({result.local_path}) -> file id {result.file_id}"
            )
        except Exception as exc:
            print(f"[warn] Google Drive upload failed for {path}: {exc}")
