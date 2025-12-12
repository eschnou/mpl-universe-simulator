"""Basic import tests to verify package structure."""


def test_import_mplsim():
    """Verify main package imports."""
    import mplsim
    assert mplsim.__version__ == "0.1.0"


def test_import_core():
    """Verify core module structure exists."""
    from mplsim import core
    assert hasattr(core, "__doc__")


def test_import_patterns():
    """Verify patterns module structure exists."""
    from mplsim import patterns
    assert hasattr(patterns, "__doc__")


def test_import_analysis():
    """Verify analysis module structure exists."""
    from mplsim import analysis
    assert hasattr(analysis, "__doc__")
