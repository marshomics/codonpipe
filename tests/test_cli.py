"""Tests for CLI interface."""

from click.testing import CliRunner

from codonpipe.cli import main


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "CodonPipe" in result.output


def test_run_help():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "genome" in result.output.lower()


def test_batch_help():
    runner = CliRunner()
    result = runner.invoke(main, ["batch", "--help"])
    assert result.exit_code == 0
    assert "batch_table" in result.output.lower()


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "codonpipe" in result.output.lower()
