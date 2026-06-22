import nox


@nox.session(python=["3.9"])
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("python", "-m", "compileall", "-q", "src")
    session.run("pytest", "-q")


@nox.session(python=["3.9"])
def lint(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "tests")
