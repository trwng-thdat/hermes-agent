"""Per-job end user: a cron job created with user_id / acting_account_id runs its turn bound to
that user, so a ``forward_session_user`` MCP server gets ``_meta["ai.jarvis/actingAccountId"]``
exactly as on a chat turn. A job without them keeps the old identity-less run (no meta)."""

import pytest

from cron.jobs import create_job, load_jobs, update_job
from cron.scheduler import _CronRunScope
from gateway.session_context import get_session_env, reset_session_vars
from tools.mcp_tool_handlers import _acting_user_meta

_META_KEY = "ai.jarvis/actingAccountId"


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate the cron store (same pattern as tests/cron/test_jobs.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path / "cron"


@pytest.fixture()
def clean_session(monkeypatch):
    for name in ("HERMES_SESSION_USER_ID", "HERMES_SESSION_ACTING_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)
    reset_session_vars()
    yield
    reset_session_vars()


def _create(**kw):
    kw.setdefault("prompt", "say hi")
    kw.setdefault("schedule", "every 1h")
    return create_job(**kw)


def _meta_during_run(job):
    """``_meta`` a forward_session_user server would get from inside the job's run scope."""
    server = type("S", (), {"_config": {"forward_session_user": True}})()
    scope = _CronRunScope(job, job["id"], "exec-1")
    scope.enter()
    try:
        return _acting_user_meta(server), get_session_env("HERMES_SESSION_USER_ID")
    finally:
        scope.exit()


class TestJobStoreActingAccount:

    def test_stored_and_round_tripped(self, tmp_cron_dir):
        job = _create(user_id=" u-1 ", acting_account_id="hd-42")
        assert job["user_id"] == "u-1"
        assert job["acting_account_id"] == "hd-42"
        assert load_jobs()[0]["acting_account_id"] == "hd-42"

    def test_absent_keeps_record_without_the_keys(self, tmp_cron_dir):
        job = _create()
        assert "user_id" not in job
        assert "acting_account_id" not in job

    def test_update_sets_and_clears(self, tmp_cron_dir):
        job = _create()
        updated = update_job(job["id"], {"user_id": "u-1", "acting_account_id": "hd-42"})
        assert updated["acting_account_id"] == "hd-42"
        cleared = update_job(job["id"], {"user_id": "", "acting_account_id": ""})
        assert cleared["user_id"] is None
        assert cleared["acting_account_id"] is None


class TestRunScopeBindsTheUser:

    def test_job_with_acting_account_sends_meta(self, tmp_cron_dir, clean_session):
        job = _create(user_id="u-1", acting_account_id="hd-42")
        meta, user_id = _meta_during_run(job)
        assert meta == {_META_KEY: "hd-42"}
        assert user_id == "u-1"

    def test_job_without_acting_account_sends_no_meta(self, tmp_cron_dir, clean_session):
        meta, user_id = _meta_during_run(_create())
        assert meta is None
        assert user_id == ""

    def test_exit_unbinds_the_user(self, tmp_cron_dir, clean_session):
        _meta_during_run(_create(user_id="u-1", acting_account_id="hd-42"))
        assert get_session_env("HERMES_SESSION_USER_ID") == ""
