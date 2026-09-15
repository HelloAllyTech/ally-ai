from unittest.mock import patch

from app.prompts.resolver import resolve_template


@patch("app.prompts.resolver.prompt_manager")
def test_resolve_template_supervisor_note_always_local(mock_prompt_manager):
    """
    Verify that the supervisor_note prompt is always loaded from local files,
    ignoring any backend version.
    """
    prompt_code = "ally_ai_shared_supervisor_note"
    internal_path = "shared/supervisor_note"
    local_template = "This is the correct local template."
    backend_prompts = {
        prompt_code: "This is a stale backend template.",
    }

    mock_prompt_manager.get_template.return_value = local_template

    # This is the broken case: backend_prompts has a stale version.
    # The resolver should ignore it and use the local version.
    result = resolve_template(
        prompt_code, backend_prompts, internal_path=internal_path
    )
    assert result == local_template, (
        "Expected the local template to be used for supervisor_note, "
        "even when a backend version is present."
    )

    # This is the working case: no backend_prompts. Should use local.
    result_no_backend = resolve_template(
        prompt_code, backend_prompts=None, internal_path=internal_path
    )
    assert result_no_backend == local_template, (
        "Expected the local template to be used for supervisor_note when "
        "no backend version is present."
    )

    # Verify that other prompts still use the backend version when available.
    other_prompt_code = "ally_ai_some_other_prompt"
    other_internal_path = "some/other_prompt"
    other_local_template = "This is another local template."
    other_backend_prompts = {
        other_prompt_code: "This is a backend template for another prompt.",
    }
    mock_prompt_manager.get_template.return_value = other_local_template

    result_other = resolve_template(
        other_prompt_code, other_backend_prompts, internal_path=other_internal_path
    )
    assert result_other == other_backend_prompts[other_prompt_code], (
        "Expected the backend template to be used for prompts other than "
        "supervisor_note."
    )
