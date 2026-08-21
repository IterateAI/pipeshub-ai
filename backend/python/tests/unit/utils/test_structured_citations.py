from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

from app.utils.structured_citations import (
    StructuredCitationLocation,
    _cell_reference_matches,
    create_resolve_structured_citations_tool,
    resolve_structured_citations,
)


def _record() -> dict:
    return {
        "id": "record-1",
        "record_name": "large.csv",
        "frontend_url": "https://app.example.com",
        "block_containers": {
            "blocks": [
                {
                    "index": 7,
                    "data": {
                        "row_number": 10196,
                        "row_values": ["27", "33324/", 87718],
                    },
                    "citation_metadata": {"row_number": 10196},
                },
                {
                    "index": 8,
                    "data": {
                        "row_number": 21216,
                        "row_values": ["55", "311942", 30795],
                    },
                    "citation_metadata": {"row_number": 21216},
                },
            ],
            "block_groups": [],
        },
    }


def _pptx_record() -> dict:
    return {
        "id": "record-pptx",
        "record_name": "slides.pptx",
        "frontend_url": "https://app.example.com",
        "block_containers": {
            "blocks": [
                {
                    "index": 16,
                    "data": "Benefits of using GitHub",
                    "citation_metadata": {"slide_number": 4},
                },
                {
                    "index": 17,
                    "data": "You can report bugs/issues you find using GitHub.",
                    "citation_metadata": {"slide_number": 4},
                },
                {
                    "index": 18,
                    "data": "Questions",
                    "citation_metadata": {"slide_number": 5},
                },
            ],
            "block_groups": [],
        },
    }


def _sqlite_record() -> dict:
    return {
        "id": "record-sqlite",
        "record_name": "metrics.sqlite",
        "frontend_url": "https://app.example.com",
        "block_containers": {
            "blocks": [
                {
                    "index": 0,
                    "data": "SQLite object cells: CREATE TABLE cells (value TEXT)",
                    "citation_metadata": {},
                },
            ],
            "block_groups": [],
        },
    }


def _sqlite_bytes(tmp_path) -> bytes:
    database = tmp_path / "metrics.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE cells (label TEXT, row_index INTEGER, value TEXT)"
        )
        connection.execute(
            "INSERT INTO cells (label, row_index, value) VALUES (?, ?, ?)",
            ("alpha", 53, "891456"),
        )
        connection.commit()
    finally:
        connection.close()
    return database.read_bytes()


def _normalized_sqlite_bytes(tmp_path) -> bytes:
    database = tmp_path / "normalized.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            'CREATE TABLE records (row_index INTEGER PRIMARY KEY, "c001" TEXT, "c007" TEXT)'
        )
        connection.execute(
            "CREATE TABLE source_schema (column_index INTEGER PRIMARY KEY, source_header TEXT)"
        )
        connection.executemany(
            "INSERT INTO source_schema (column_index, source_header) VALUES (?, ?)",
            [(1, "uscode"), (7, "qp1")],
        )
        connection.execute(
            'INSERT INTO records (row_index, "c001", "c007") VALUES (?, ?, ?)',
            (2556, "98", "6433"),
        )
        connection.commit()
    finally:
        connection.close()
    return database.read_bytes()


def test_resolves_csv_rows_to_real_refs() -> None:
    mapper = MagicMock()
    mapper.get_or_create_ref.side_effect = ["ref1", "ref2"]

    result = resolve_structured_citations(
        record_id="record-1",
        locations=[
            StructuredCitationLocation(csv_row=10196),
            StructuredCitationLocation(csv_row=21216),
        ],
        virtual_record_id_to_result={"vr-1": _record()},
        ref_mapper=mapper,
    )

    assert result["ok"] is True
    assert [item["citation_markdown"] for item in result["citations"]] == [
        "[source](ref1)",
        "[source](ref2)",
    ]
    assert result["citations"][0]["location"] == "CSV row 10196"
    mapper.get_or_create_ref.assert_any_call(
        "https://app.example.com/record/record-1/preview#blockIndex=7"
    )


def test_unresolved_location_does_not_allocate_ref() -> None:
    mapper = MagicMock()

    result = resolve_structured_citations(
        record_id="record-1",
        locations=[StructuredCitationLocation(csv_row=99999)],
        virtual_record_id_to_result={"vr-1": _record()},
        ref_mapper=mapper,
    )

    assert result["ok"] is False
    assert result["unresolved_locations"] == [{"csv_row": 99999}]
    mapper.get_or_create_ref.assert_not_called()


def test_slide_location_returns_every_parsed_block_on_slide() -> None:
    mapper = MagicMock()
    mapper.get_or_create_ref.side_effect = ["ref1", "ref2"]

    result = resolve_structured_citations(
        record_id="record-pptx",
        locations=[StructuredCitationLocation(slide_number=4)],
        virtual_record_id_to_result={"vr-1": _pptx_record()},
        ref_mapper=mapper,
    )

    assert result["ok"] is True
    assert [item["block_index"] for item in result["citations"]] == [16, 17]
    assert [item["citation_markdown"] for item in result["citations"]] == [
        "[source](ref1)",
        "[source](ref2)",
    ]


def test_single_excel_cell_matches_parsed_row_range() -> None:
    assert _cell_reference_matches("C7", "A7:H7") is True
    assert _cell_reference_matches("C8", "A7:H7") is False


def test_resolves_exact_sqlite_value_to_materialized_block(tmp_path) -> None:
    mapper = MagicMock()
    mapper.get_or_create_ref.return_value = "ref-sqlite"
    record = _sqlite_record()

    result = resolve_structured_citations(
        record_id="record-sqlite",
        locations=[
            StructuredCitationLocation(
                sqlite_table="cells",
                sqlite_rowid=1,
                sqlite_column="value",
            )
        ],
        virtual_record_id_to_result={"vr-sqlite": record},
        ref_mapper=mapper,
        attachment_input_files={"metrics.sqlite": _sqlite_bytes(tmp_path)},
    )

    assert result["ok"] is True
    assert result["citations"] == [
        {
            "citation_id": "ref-sqlite",
            "citation_markdown": "[source](ref-sqlite)",
            "location": "SQLite table cells, rowid 1, column value",
            "block_index": 1,
            "content_preview": "SQLite table cells, rowid 1, column value: 891456",
        }
    ]
    assert record["block_containers"]["blocks"][1]["data"].endswith(": 891456")
    mapper.get_or_create_ref.assert_called_once_with(
        "https://app.example.com/record/record-sqlite/preview#blockIndex=1"
    )


def test_resolves_sqlite_value_by_exact_filters_and_virtual_record_id(tmp_path) -> None:
    mapper = MagicMock()
    mapper.get_or_create_ref.return_value = "ref-filtered"
    record = _sqlite_record()

    result = resolve_structured_citations(
        record_id="vr-sqlite",
        locations=[
            StructuredCitationLocation(
                sqlite_table="cells",
                sqlite_column="value",
                sqlite_filters={"label": "alpha", "row_index": 53},
            )
        ],
        virtual_record_id_to_result={"vr-sqlite": record},
        ref_mapper=mapper,
        attachment_input_files={"metrics.sqlite": _sqlite_bytes(tmp_path)},
    )

    assert result["ok"] is True
    assert result["citations"][0]["content_preview"].endswith(": 891456")
    metadata = record["block_containers"]["blocks"][1]["citation_metadata"]
    assert metadata["sqlite_rowid"] == 1
    assert metadata["sqlite_filters"] == {"label": "alpha", "row_index": 53}


def test_resolves_sqlite_filename_and_source_header_aliases(tmp_path) -> None:
    mapper = MagicMock()
    mapper.get_or_create_ref.return_value = "ref-normalized"
    record = _sqlite_record()
    record["record_name"] = "normalized.sqlite"

    result = resolve_structured_citations(
        record_id="normalized.sqlite",
        locations=[
            StructuredCitationLocation(
                sqlite_table="records",
                sqlite_column="qp1",
                sqlite_filters={"row_index": 2556, "uscode": "98"},
            )
        ],
        virtual_record_id_to_result={"vr-normalized": record},
        ref_mapper=mapper,
        attachment_input_files={
            "normalized.sqlite": _normalized_sqlite_bytes(tmp_path)
        },
    )

    assert result["ok"] is True
    assert result["citations"][0]["content_preview"].endswith("column qp1: 6433")
    metadata = record["block_containers"]["blocks"][1]["citation_metadata"]
    assert metadata["sqlite_column"] == "qp1"
    assert metadata["sqlite_physical_column"] == "c007"
    assert metadata["sqlite_physical_filters"] == {
        "row_index": 2556,
        "c001": "98",
    }


def test_sqlite_filename_resolution_rejects_ambiguous_records(tmp_path) -> None:
    mapper = MagicMock()
    first = _sqlite_record()
    second = _sqlite_record()
    second["id"] = "record-sqlite-2"

    result = resolve_structured_citations(
        record_id="metrics.sqlite",
        locations=[
            StructuredCitationLocation(
                sqlite_table="cells",
                sqlite_rowid=1,
                sqlite_column="value",
            )
        ],
        virtual_record_id_to_result={"vr-1": first, "vr-2": second},
        ref_mapper=mapper,
        attachment_input_files={"metrics.sqlite": _sqlite_bytes(tmp_path)},
    )

    assert result["ok"] is False
    assert result["error"] == "Record ID is not available in this conversation."
    mapper.get_or_create_ref.assert_not_called()


def test_sqlite_resolution_rejects_unknown_column(tmp_path) -> None:
    mapper = MagicMock()
    record = _sqlite_record()

    result = resolve_structured_citations(
        record_id="record-sqlite",
        locations=[
            StructuredCitationLocation(
                sqlite_table="cells",
                sqlite_rowid=1,
                sqlite_column="missing",
            )
        ],
        virtual_record_id_to_result={"vr-sqlite": record},
        ref_mapper=mapper,
        attachment_input_files={"metrics.sqlite": _sqlite_bytes(tmp_path)},
    )

    assert result["ok"] is False
    assert result["unresolved_locations"] == [
        {"sqlite_table": "cells", "sqlite_rowid": 1, "sqlite_column": "missing"}
    ]
    mapper.get_or_create_ref.assert_not_called()


def test_sqlite_resolution_rejects_unknown_filter_column(tmp_path) -> None:
    mapper = MagicMock()

    result = resolve_structured_citations(
        record_id="vr-sqlite",
        locations=[
            StructuredCitationLocation(
                sqlite_table="cells",
                sqlite_column="value",
                sqlite_filters={"missing": "alpha"},
            )
        ],
        virtual_record_id_to_result={"vr-sqlite": _sqlite_record()},
        ref_mapper=mapper,
        attachment_input_files={"metrics.sqlite": _sqlite_bytes(tmp_path)},
    )

    assert result["ok"] is False
    mapper.get_or_create_ref.assert_not_called()


async def test_tool_factory_uses_injected_mapper_and_records() -> None:
    mapper = MagicMock()
    mapper.get_or_create_ref.return_value = "ref9"
    citation_tool = create_resolve_structured_citations_tool(
        {"vr-1": _record()},
        mapper,
    )

    result = await citation_tool.ainvoke(
        {
            "record_id": "record-1",
            "locations": [{"csv_row": 10196}],
        }
    )

    assert result["ok"] is True
    assert result["citations"][0]["citation_markdown"] == "[source](ref9)"


@patch("app.modules.agents.qna.tool_system._global_tools_registry")
def test_react_tool_loader_binds_resolver_after_attachment_resolution(
    registry: MagicMock,
) -> None:
    from app.modules.agents.qna.tool_system import get_agent_tools_with_schemas

    registry.get_all_tools.return_value = {}
    state = {
        "has_knowledge": False,
        "virtual_record_id_to_result": {"vr-1": _record()},
        "citation_ref_mapper": MagicMock(),
    }

    tools = get_agent_tools_with_schemas(state)

    assert "resolve_structured_citations" in [tool.name for tool in tools]
