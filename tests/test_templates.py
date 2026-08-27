"""Templates are data, and the rest of the system leans on their shape."""

from __future__ import annotations

import pytest

from app.templates import DEFAULT_TEMPLATE_ID, TEMPLATES, get_template, list_templates


def test_the_default_template_matches_the_paper_agenda():
    template = get_template(None)
    assert template.id == DEFAULT_TEMPLATE_ID
    assert [(s.title, s.minutes) for s in template.agenda] == [
        ("Open", 2), ("Goals", 5), ("Reality", 4), ("Way Forward", 4)
    ]
    assert template.planned_minutes == 15


def test_every_template_is_internally_consistent():
    for template in list_templates():
        assert template.agenda, "a template needs at least one agenda item"
        section_ids = [s.id for s in template.sections]
        assert len(section_ids) == len(set(section_ids)), f"{template.id} has duplicate section ids"
        for section in template.sections:
            field_ids = [f.id for f in section.fields]
            assert field_ids, f"{template.id}.{section.id} has no fields"
            assert len(field_ids) == len(set(field_ids))
            for field in section.fields:
                assert field.kind in ("text", "list", "actions", "choice", "ratio")
                if field.kind == "choice":
                    assert field.choices, f"{field.id} is a choice with no choices"


def test_agenda_items_carry_cues_for_the_offline_aligner():
    for template in list_templates():
        for section in template.agenda:
            assert section.cues, f"{template.id}.{section.id} has no cues"


def test_record_sections_are_excluded_from_the_agenda():
    template = get_template(None)
    assert "record" in {s.id for s in template.record_sections}
    assert "record" not in {s.id for s in template.agenda}


def test_lookup_helpers():
    template = get_template(None)
    assert template.section("goals").title == "Goals"
    assert template.section("nope") is None
    assert template.field("goals", "strength_named").kind == "text"
    assert template.field("goals", "nope") is None
    assert template.field("nope", "x") is None


def test_an_unknown_template_id_raises():
    with pytest.raises(KeyError):
        get_template("not-a-template")


def test_templates_are_registered_under_their_own_id():
    for key, template in TEMPLATES.items():
        assert key == template.id
