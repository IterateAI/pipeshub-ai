from __future__ import annotations

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
                    "data": {"row_number": 10196, "row_values": ["27", "33324/", 87718]},
                    "citation_metadata": {"row_number": 10196},
                },
                {
                    "index": 8,
                    "data": {"row_number": 21216, "row_values": ["55", "311942", 30795]},
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
