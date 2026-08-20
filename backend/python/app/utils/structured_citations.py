from __future__ import annotations

import re
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
    return supplied


def _record_extension(record: dict[str, Any]) -> str:
    name = str(record.get("record_name") or "")
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


def resolve_structured_citations(
    *,
    record_id: str,
    locations: list[StructuredCitationLocation],
    virtual_record_id_to_result: dict[str, Any],
    ref_mapper: CitationRefMapper,
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
            for candidate in virtual_record_id_to_result.values()
            if isinstance(candidate, dict)
            and str(
                candidate.get("id")
                or candidate.get("record_id")
                or candidate.get("_key")
                or ""
            )
            == record_id
        ),
        None,
    )
    if record is None:
        return {
            "ok": False,
            "error": "Record ID is not available in this conversation.",
        }

    extension = _record_extension(record)
    entries = structured_record_block_entries(record, extension)
    citations: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen_blocks: set[int] = set()

    for requested in locations:
        match = next(
            (entry for entry in entries if _location_matches(requested, entry)),
            None,
        )
        if match is None:
            unresolved.append(requested.model_dump(exclude_none=True))
            continue
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
) -> Callable[..., Any]:
    @tool(
        "resolve_structured_citations",
        args_schema=ResolveStructuredCitationsArgs,
    )
    async def resolve_structured_citations_tool(
        record_id: str,
        locations: list[StructuredCitationLocation],
    ) -> dict[str, Any]:
        """Resolve exact rows, worksheet cells, slides, or blocks to real citations.

        Use this after inspecting an oversized structured attachment in the coding
        sandbox. Never guess citation IDs. Pass the exact Record ID from the
        attachment context and only source locations that support the final answer.
        """
        return resolve_structured_citations(
            record_id=record_id,
            locations=locations,
            virtual_record_id_to_result=virtual_record_id_to_result,
            ref_mapper=ref_mapper,
        )

    return resolve_structured_citations_tool
