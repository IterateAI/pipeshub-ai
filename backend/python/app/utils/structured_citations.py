from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.attachment_utils import structured_record_block_entries

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.utils.chat_helpers import CitationRefMapper


class StructuredCitationLocation(BaseModel):
    block_index: int | None = Field(
        default=None,
        description="Exact parsed block index, when known.",
    )
    csv_row: int | None = Field(
        default=None,
        description="One-based CSV file row number, including the header row.",
    )
    sheet_name: str | None = Field(
        default=None,
        description="Exact Excel worksheet name.",
    )
    cell_reference: str | None = Field(
        default=None,
        description="Excel cell or row range such as B7 or A7:H7.",
    )
    worksheet_row: int | None = Field(
        default=None,
        description="Excel worksheet row number when an exact cell is not available.",
    )
    slide_number: int | None = Field(
        default=None,
        description="One-based PowerPoint slide number.",
    )
    sqlite_table: str | None = Field(
        default=None,
        description="Exact SQLite table name inspected in the sandbox.",
    )
    sqlite_rowid: int | None = Field(
        default=None,
        description="Exact SQLite rowid inspected in the sandbox.",
    )
    sqlite_column: str | None = Field(
        default=None,
        description="Exact SQLite column containing the cited value.",
    )
    sqlite_filters: dict[str, str | int | float | bool | None] | None = Field(
        default=None,
        description=(
            "Exact SQLite equality filters used to identify one inspected row, "
            "for example {'sheet_name': 'LA', 'cell_coordinate': 'D97'}. "
            "Use this instead of guessing sqlite_rowid."
        ),
    )


class ResolveStructuredCitationsArgs(BaseModel):
    record_id: str = Field(
        ...,
        description="Exact Record ID shown in the oversized attachment context.",
    )
    locations: list[StructuredCitationLocation] = Field(
        ...,
        min_length=1,
        description=(
            "Exact source locations found while inspecting the file in the sandbox. "
            "Provide only locations that directly support the final answer."
        ),
    )


_CELL_RE = re.compile(r"^([A-Z]+)([0-9]+)$", re.IGNORECASE)
_CELL_RANGE_RE = re.compile(
    r"^([A-Z]+)([0-9]+):([A-Z]+)([0-9]+)$",
    re.IGNORECASE,
)
_SQLITE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQLITE_EXTENSIONS = {"sqlite", "sqlite3", "db"}


def _column_number(label: str) -> int:
    value = 0
    for char in label.upper():
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def _cell_reference_matches(requested: str, available: str) -> bool:
    requested = requested.strip().replace("$", "").upper()
    available = available.strip().replace("$", "").upper()
    if requested == available:
        return True
    requested_cell = _CELL_RE.fullmatch(requested)
    available_range = _CELL_RANGE_RE.fullmatch(available)
    if not requested_cell or not available_range:
        return False
    requested_col = _column_number(requested_cell.group(1))
    requested_row = int(requested_cell.group(2))
    start_col = _column_number(available_range.group(1))
    start_row = int(available_range.group(2))
    end_col = _column_number(available_range.group(3))
    end_row = int(available_range.group(4))
    return (
        min(start_col, end_col) <= requested_col <= max(start_col, end_col)
        and min(start_row, end_row) <= requested_row <= max(start_row, end_row)
    )


def _location_matches(
    requested: StructuredCitationLocation,
    entry: dict[str, Any],
) -> bool:
    supplied = False
    if requested.block_index is not None:
        supplied = True
        if entry["block_index"] != requested.block_index:
            return False
    if requested.csv_row is not None:
        supplied = True
        if entry["csv_row"] != requested.csv_row:
            return False
    if requested.sheet_name is not None:
        supplied = True
        if (entry["sheet_name"] or "").casefold() != requested.sheet_name.casefold():
            return False
    if requested.cell_reference is not None:
        supplied = True
        available = entry["cell_reference"]
        if not available or not _cell_reference_matches(
            requested.cell_reference, available
        ):
            return False
    if requested.worksheet_row is not None:
        supplied = True
        if entry["worksheet_row"] != requested.worksheet_row:
            return False
    if requested.slide_number is not None:
        supplied = True
        if entry["slide_number"] != requested.slide_number:
            return False
    if requested.sqlite_table is not None:
        supplied = True
        if (entry.get("sqlite_table") or "").casefold() != requested.sqlite_table.casefold():
            return False
    if requested.sqlite_rowid is not None:
        supplied = True
        if entry.get("sqlite_rowid") != requested.sqlite_rowid:
            return False
    if requested.sqlite_column is not None:
        supplied = True
        if (entry.get("sqlite_column") or "").casefold() != requested.sqlite_column.casefold():
            return False
    if requested.sqlite_filters:
        supplied = True
        if (entry.get("sqlite_filters") or {}) != requested.sqlite_filters:
            return False
    return supplied


def _record_extension(record: dict[str, Any]) -> str:
    name = str(record.get("record_name") or record.get("recordName") or "")
    if "." in name:
        return name.rsplit(".", 1)[-1].lower()
    mime_type = str(record.get("mime_type") or "").lower()
    mime_extensions = {
        "text/csv": "csv",
        "application/csv": "csv",
        "application/vnd.ms-excel": "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-powerpoint": "ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    }
    return mime_extensions.get(mime_type, "")


def _attachment_bytes(
    record: dict[str, Any],
    attachment_input_files: dict[str, bytes] | None,
) -> bytes | None:
    if not attachment_input_files:
        return None
    record_name = str(record.get("record_name") or record.get("recordName") or "")
    candidates = (record_name, Path(record_name).name)
    return next(
        (
            attachment_input_files[name]
            for name in candidates
            if name and isinstance(attachment_input_files.get(name), bytes)
        ),
        None,
    )


def _add_sqlite_value_blocks(
    record: dict[str, Any],
    locations: list[StructuredCitationLocation],
    attachment_input_files: dict[str, bytes] | None,
) -> None:
    """Materialize exact, read-only SQLite values as citable in-memory blocks."""
    raw_database = _attachment_bytes(record, attachment_input_files)
    if raw_database is None:
        return

    requested = [
        location
        for location in locations
        if location.sqlite_table
        and (location.sqlite_rowid is not None or location.sqlite_filters)
        and location.sqlite_column
    ]
    if not requested:
        return

    block_container = record.setdefault("block_containers", {})
    blocks = block_container.setdefault("blocks", [])
    existing = {
        (
            str((block.get("citation_metadata") or {}).get("sqlite_table") or "").casefold(),
            (block.get("citation_metadata") or {}).get("sqlite_rowid"),
            str((block.get("citation_metadata") or {}).get("sqlite_column") or "").casefold(),
            tuple(
                sorted(
                    ((block.get("citation_metadata") or {}).get("sqlite_filters") or {}).items()
                )
            ),
        )
        for block in blocks
        if isinstance(block, dict)
    }
    next_index = max(
        (int(block.get("index") or 0) for block in blocks if isinstance(block, dict)),
        default=-1,
    ) + 1
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as temporary:
        temporary.write(raw_database)
        temporary.flush()
        connection = sqlite3.connect(f"file:{temporary.name}?mode=ro", uri=True)
        try:
            for location in requested:
                table = str(location.sqlite_table)
                column = str(location.sqlite_column)
                if not _SQLITE_IDENTIFIER_RE.fullmatch(table) or not _SQLITE_IDENTIFIER_RE.fullmatch(column):
                    continue
                columns = {
                    str(row[1]).casefold()
                    for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
                }
                if column.casefold() not in columns:
                    continue

                filters = location.sqlite_filters or {}
                if filters:
                    if any(
                        not _SQLITE_IDENTIFIER_RE.fullmatch(filter_column)
                        or filter_column.casefold() not in columns
                        for filter_column in filters
                    ):
                        continue
                    predicates: list[str] = []
                    parameters: list[Any] = []
                    for filter_column, filter_value in filters.items():
                        if filter_value is None:
                            predicates.append(f'"{filter_column}" IS NULL')
                        else:
                            predicates.append(f'"{filter_column}" = ?')
                            parameters.append(filter_value)
                    rows = connection.execute(
                        f'SELECT rowid, "{column}" FROM "{table}" '
                        f'WHERE {" AND ".join(predicates)} LIMIT 2',
                        parameters,
                    ).fetchall()
                else:
                    rows = connection.execute(
                        f'SELECT rowid, "{column}" FROM "{table}" WHERE rowid = ?',
                        (location.sqlite_rowid,),
                    ).fetchall()
                if len(rows) != 1:
                    continue
                actual_rowid, value = rows[0]
                location.sqlite_rowid = int(actual_rowid)
                key = (
                    table.casefold(),
                    location.sqlite_rowid,
                    column.casefold(),
                    tuple(sorted(filters.items())),
                )
                if key in existing:
                    continue
                blocks.append(
                    {
                        "index": next_index,
                        "type": "text",
                        "format": "txt",
                        "data": (
                            f"SQLite table {table}, rowid {location.sqlite_rowid}, "
                            f"column {column}: {value}"
                        ),
                        "citation_metadata": {
                            "section_title": f"SQLite row: {table} rowid {location.sqlite_rowid}",
                            "sqlite_table": table,
                            "sqlite_rowid": location.sqlite_rowid,
                            "sqlite_column": column,
                            "sqlite_filters": filters,
                        },
                    }
                )
                existing.add(key)
                next_index += 1
        finally:
            connection.close()


def resolve_structured_citations(
    *,
    record_id: str,
    locations: list[StructuredCitationLocation],
    virtual_record_id_to_result: dict[str, Any],
    ref_mapper: CitationRefMapper,
    attachment_input_files: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    locations = [
        location
        if isinstance(location, StructuredCitationLocation)
        else StructuredCitationLocation.model_validate(location)
        for location in locations
    ]
    record = next(
        (
            candidate
            for virtual_record_id, candidate in virtual_record_id_to_result.items()
            if isinstance(candidate, dict)
            and record_id
            in {
                str(virtual_record_id),
                str(candidate.get("id") or ""),
                str(candidate.get("record_id") or ""),
                str(candidate.get("_key") or ""),
            }
        ),
        None,
    )
    if record is None:
        return {
            "ok": False,
            "error": "Record ID is not available in this conversation.",
        }

    extension = _record_extension(record)
    if extension in _SQLITE_EXTENSIONS:
        _add_sqlite_value_blocks(
            record,
            locations,
            attachment_input_files,
        )
    entries = structured_record_block_entries(record, extension)
    citations: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen_blocks: set[int] = set()

    for requested in locations:
        matches = [
            entry for entry in entries if _location_matches(requested, entry)
        ]
        if not matches:
            unresolved.append(requested.model_dump(exclude_none=True))
            continue
        for match in matches:
            if match["block_index"] in seen_blocks:
                continue
            seen_blocks.add(match["block_index"])
            citation_id = ref_mapper.get_or_create_ref(match["block_url"])
            citations.append(
                {
                    "citation_id": citation_id,
                    "citation_markdown": f"[source]({citation_id})",
                    "location": match["location"],
                    "block_index": match["block_index"],
                    "content_preview": match["rendered_data"][:500],
                }
            )

    if not citations:
        return {
            "ok": False,
            "error": "None of the requested locations matched parsed source blocks.",
            "unresolved_locations": unresolved,
            "available_location_examples": [
                entry["location"] for entry in entries[:10]
            ],
        }
    return {
        "ok": True,
        "record_id": record_id,
        "citations": citations,
        "unresolved_locations": unresolved,
        "instruction": (
            "Use each citation_markdown value exactly after the claim it supports."
        ),
    }


def create_resolve_structured_citations_tool(
    virtual_record_id_to_result: dict[str, Any],
    ref_mapper: CitationRefMapper,
    attachment_input_files: dict[str, bytes] | None = None,
) -> Callable[..., Any]:
    @tool(
        "resolve_structured_citations",
        args_schema=ResolveStructuredCitationsArgs,
    )
    async def resolve_structured_citations_tool(
        record_id: str,
        locations: list[StructuredCitationLocation],
    ) -> dict[str, Any]:
        """Resolve exact rows, worksheet cells, SQLite values, slides, or blocks to real citations.

        Use this after inspecting an oversized structured attachment in the coding
        sandbox. Never guess citation IDs. Pass the exact Record ID from the
        attachment context and only source locations that support the final answer.
        For SQLite, pass sqlite_table and sqlite_column plus either the exact
        sqlite_rowid or exact sqlite_filters from the inspected query. Never guess
        rowids.
        """
        return resolve_structured_citations(
            record_id=record_id,
            locations=locations,
            virtual_record_id_to_result=virtual_record_id_to_result,
            ref_mapper=ref_mapper,
            attachment_input_files=attachment_input_files,
        )

    return resolve_structured_citations_tool
