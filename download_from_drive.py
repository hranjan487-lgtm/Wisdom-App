"""
Download all PDFs from a Google Drive folder using a service account
(authenticated Drive API access — reliable, not subject to the rate limits
that anonymous "anyone with link" downloads hit).

Usage:
    python download_from_drive.py --folder-id FOLDER_ID --out-dir pdfs

Requires the GOOGLE_SERVICE_ACCOUNT_JSON environment variable to contain
the full JSON key content (not a file path — the raw JSON text).
"""

import argparse
import io
import json
import os
import re
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def sanitize_filename(name):
    """Make a Drive file title safe to use as a local filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder-id", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="pdfs")
    args = ap.parse_args()

    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise SystemExit("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")

    creds_info = json.loads(creds_json)
    credentials = service_account.Credentials.from_service_account_info(
        creds_info, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # List every PDF in the folder, paging through results
    files = []
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{args.folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="nextPageToken, files(id, name, size)",
            pageToken=page_token,
            pageSize=100,
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    print(f"Found {len(files)} PDF files in the folder")

    downloaded = 0
    failed = []
    for i, f in enumerate(files, 1):
        filename = sanitize_filename(f["name"])
        dest_path = out_dir / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            downloaded += 1
            continue  # already downloaded (supports safe re-runs)

        print(f"[{i}/{len(files)}] Downloading: {f['name']}")
        try:
            request = service.files().get_media(fileId=f["id"])
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            dest_path.write_bytes(buffer.getvalue())
            downloaded += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(f["name"])

    print(f"\nDownloaded {downloaded}/{len(files)} files to {out_dir}")
    if failed:
        print(f"{len(failed)} files failed:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
