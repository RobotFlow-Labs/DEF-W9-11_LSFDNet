import pytest


def test_create_app() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from anima_lsfdnet.api import create_app

    app = create_app()
    assert app.title == "LSFDNet API"
