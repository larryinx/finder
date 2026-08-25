"""Plugin registry contract."""

import pytest

from search_ttl.trainer.registry import (
    EXPERIENCES,
    CURATORS,
    REFLECTORS,
    REWARDS,
    import_default_plugins,
    register_experience,
    register_reflector,
)


def test_default_plugins_register():
    import_default_plugins()
    assert "search_agent_lasj" in REWARDS
    assert "qa_em_f1" in REWARDS
    assert "querychain_tactic" in REFLECTORS
    assert "querychain_staged" in REFLECTORS
    assert "react_tactic" in CURATORS
    assert "react_staged" in CURATORS
    assert "tactic_playbook" in EXPERIENCES
    assert "staged_playbook" in EXPERIENCES


def test_double_register_raises():
    @register_reflector("test_double_unique")
    class A:  # noqa: D401
        pass

    with pytest.raises(ValueError):
        @register_reflector("test_double_unique")
        class B:  # noqa: D401
            pass


def test_register_idempotent_for_same_class():
    @register_experience("test_idem_unique")
    class C:  # noqa: D401
        pass

    # Re-registering the *same* class is a no-op (avoids errors during repeated
    # `import_default_plugins()` calls in long-running scripts).
    register_experience("test_idem_unique")(C)
