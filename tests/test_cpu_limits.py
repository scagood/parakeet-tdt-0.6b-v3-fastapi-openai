"""cgroup CPU quota detection.

A Kubernetes CPU limit is a CFS quota rather than a cpuset, so it is invisible
to sched_getaffinity/psutil. These tests pin the parsing of both cgroup
layouts, since misreading a quota silently oversubscribes ORT's thread pools.
"""
from __future__ import annotations

from parakeet_service.config import cgroup_cpu_limit


def _write_v2(root, contents):
    (root / "cpu.max").write_text(contents)


def _write_v1(root, quota, period):
    cpu_dir = root / "cpu"
    cpu_dir.mkdir()
    (cpu_dir / "cpu.cfs_quota_us").write_text(quota)
    (cpu_dir / "cpu.cfs_period_us").write_text(period)


def test_missing_cgroup_files_report_no_limit(tmp_path):
    assert cgroup_cpu_limit(tmp_path) is None


def test_v2_unlimited_quota(tmp_path):
    _write_v2(tmp_path, "max 100000")
    assert cgroup_cpu_limit(tmp_path) is None


def test_v2_whole_core_quota(tmp_path):
    _write_v2(tmp_path, "400000 100000")
    assert cgroup_cpu_limit(tmp_path) == 4


def test_v2_fractional_quota_rounds_up(tmp_path):
    # 3.5 cores: 4 threads timeshare the same quota without oversubscribing it.
    _write_v2(tmp_path, "350000 100000")
    assert cgroup_cpu_limit(tmp_path) == 4


def test_v2_sub_core_quota_floors_at_one(tmp_path):
    _write_v2(tmp_path, "50000 100000")
    assert cgroup_cpu_limit(tmp_path) == 1


def test_v2_trailing_newline_is_tolerated(tmp_path):
    _write_v2(tmp_path, "200000 100000\n")
    assert cgroup_cpu_limit(tmp_path) == 2


def test_v2_malformed_contents_report_no_limit(tmp_path):
    _write_v2(tmp_path, "not-a-quota 100000")
    assert cgroup_cpu_limit(tmp_path) is None


def test_v1_quota_is_read(tmp_path):
    _write_v1(tmp_path, "400000", "100000")
    assert cgroup_cpu_limit(tmp_path) == 4


def test_v1_unlimited_quota(tmp_path):
    _write_v1(tmp_path, "-1", "100000")
    assert cgroup_cpu_limit(tmp_path) is None


def test_v1_zero_period_reports_no_limit(tmp_path):
    _write_v1(tmp_path, "400000", "0")
    assert cgroup_cpu_limit(tmp_path) is None


def test_v2_takes_precedence_over_v1(tmp_path):
    _write_v2(tmp_path, "200000 100000")
    _write_v1(tmp_path, "800000", "100000")
    assert cgroup_cpu_limit(tmp_path) == 2
