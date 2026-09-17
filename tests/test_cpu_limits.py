"""cgroup CPU quota detection.

A Kubernetes CPU limit is a CFS quota rather than a cpuset, so it is invisible
to sched_getaffinity/psutil. These tests pin the parsing of both cgroup
layouts, since misreading a quota silently oversubscribes ORT's thread pools.
"""
from __future__ import annotations

from parakeet_service.config import cgroup_cpu_limit, effective_cpu_counts


def _write_v2(root, contents, path=""):
    cgroup = root / path
    cgroup.mkdir(parents=True, exist_ok=True)
    (cgroup / "cpu.max").write_text(contents)


def _write_v1(root, quota, period, path="", controller="cpu"):
    cpu_dir = root / controller / path
    cpu_dir.mkdir(parents=True, exist_ok=True)
    (cpu_dir / "cpu.cfs_quota_us").write_text(quota)
    (cpu_dir / "cpu.cfs_period_us").write_text(period)


def _write_proc(tmp_path, contents):
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text(contents)
    return proc


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


# --- resolving the process's own cgroup --------------------------------------
# The mount root is only the process's cgroup inside a private cgroup
# namespace. Under systemd CPUQuota= or --cgroupns=host the process sits
# several levels down, and reading the root alone silently misses the limit.

def test_v2_quota_on_own_nested_cgroup_is_found(tmp_path):
    root = tmp_path / "sys"
    _write_v2(root, "max 100000")
    _write_v2(root, "200000 100000", "system.slice/parakeet.service")
    proc = _write_proc(tmp_path, "0::/system.slice/parakeet.service\n")
    assert cgroup_cpu_limit(root, proc) == 2


def test_v2_quota_on_an_ancestor_cgroup_applies(tmp_path):
    # Kubernetes sets the limit on the container cgroup; --cgroupns=host puts
    # the process one level below it.
    root = tmp_path / "sys"
    _write_v2(root, "400000 100000", "kubepods/pod1/container")
    _write_v2(root, "max 100000", "kubepods/pod1/container/leaf")
    proc = _write_proc(tmp_path, "0::/kubepods/pod1/container/leaf\n")
    assert cgroup_cpu_limit(root, proc) == 4


def test_v2_tightest_nested_quota_wins(tmp_path):
    root = tmp_path / "sys"
    _write_v2(root, "800000 100000", "a")
    _write_v2(root, "200000 100000", "a/b")
    _write_v2(root, "600000 100000", "a/b/c")
    proc = _write_proc(tmp_path, "0::/a/b/c\n")
    assert cgroup_cpu_limit(root, proc) == 2


def test_v2_unlimited_at_every_level_reports_no_limit(tmp_path):
    root = tmp_path / "sys"
    _write_v2(root, "max 100000")
    _write_v2(root, "max 100000", "a/b")
    proc = _write_proc(tmp_path, "0::/a/b\n")
    assert cgroup_cpu_limit(root, proc) is None


def test_v1_quota_on_own_nested_cgroup_is_found(tmp_path):
    root = tmp_path / "sys"
    _write_v1(root, "-1", "100000")
    _write_v1(root, "300000", "100000", "docker/abc123")
    proc = _write_proc(
        tmp_path,
        "12:memory:/docker/abc123\n3:cpu,cpuacct:/docker/abc123\n1:name=systemd:/\n",
    )
    assert cgroup_cpu_limit(root, proc) == 3


def test_v1_co_mounted_cpu_cpuacct_directory_is_read(tmp_path):
    root = tmp_path / "sys"
    _write_v1(root, "200000", "100000", "docker/abc123", controller="cpu,cpuacct")
    proc = _write_proc(tmp_path, "3:cpu,cpuacct:/docker/abc123\n")
    assert cgroup_cpu_limit(root, proc) == 2


def test_v1_controller_directory_named_as_proc_lists_it(tmp_path):
    # Some hosts spell the co-mount "cpuacct,cpu" with no "cpu" symlink; the
    # controller string from /proc/self/cgroup is the directory name there.
    root = tmp_path / "sys"
    _write_v1(root, "200000", "100000", "docker/abc123", controller="cpuacct,cpu")
    proc = _write_proc(tmp_path, "3:cpuacct,cpu:/docker/abc123\n")
    assert cgroup_cpu_limit(root, proc) == 2


def test_v1_tightest_nested_quota_wins(tmp_path):
    root = tmp_path / "sys"
    _write_v1(root, "100000", "100000", "a")
    _write_v1(root, "-1", "100000", "a/b")
    proc = _write_proc(tmp_path, "1:cpu:/a/b\n")
    assert cgroup_cpu_limit(root, proc) == 1


def test_hybrid_layout_falls_back_to_v1_when_unified_has_no_cpu_files(tmp_path):
    # Hybrid hosts list both "0::/" and a v1 cpu hierarchy; the unified mount
    # carries no cpu.max there, so the v1 quota must still be read.
    root = tmp_path / "sys"
    _write_v1(root, "200000", "100000", "docker/abc123")
    proc = _write_proc(tmp_path, "0::/\n1:cpu:/docker/abc123\n")
    assert cgroup_cpu_limit(root, proc) == 2


def test_missing_proc_cgroup_falls_back_to_the_mount_root(tmp_path):
    _write_v2(tmp_path, "400000 100000")
    assert cgroup_cpu_limit(tmp_path, tmp_path / "missing") == 4


def test_malformed_proc_cgroup_lines_are_ignored(tmp_path):
    root = tmp_path / "sys"
    _write_v2(root, "400000 100000")
    proc = _write_proc(tmp_path, "garbage\n\n0::/\n")
    assert cgroup_cpu_limit(root, proc) == 4


# --- clamping ---------------------------------------------------------------
# The quota only matters once it is applied to the detected counts, and the
# module-level wiring runs at import against the real /sys/fs/cgroup, so the
# clamp is exercised here rather than through config's own constants.

def test_no_quota_leaves_detection_untouched():
    assert effective_cpu_counts(8, 16, None) == (8, 16)


def test_quota_clamps_both_counts():
    # The case this guards: a 4-core pod on a 64-core node.
    assert effective_cpu_counts(64, 64, 4) == (4, 4)


def test_quota_above_detection_does_not_inflate():
    assert effective_cpu_counts(4, 8, 32) == (4, 8)


def test_physical_never_exceeds_logical_after_clamping():
    physical, logical = effective_cpu_counts(64, 64, 2)
    assert physical <= logical


def test_clamped_counts_never_reach_zero():
    # cgroup_cpu_limit() reports None rather than 0 for an absent limit, so a
    # quota that reaches here is always >= 1; the floor guards the arithmetic
    # regardless.
    assert effective_cpu_counts(64, 64, 1) == (1, 1)
