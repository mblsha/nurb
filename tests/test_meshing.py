"""Precision must be explicit, bounded, and independent of the preview mesh."""
import time

import pytest
from build123d import Box, Sphere

from nurb import builder
from nurb.meshing import MeshingError, VerificationPolicy, display_provenance, verification_mesh


def test_absolute_mesh_uses_a_separate_shape_and_reports_requested_not_certified_accuracy():
    shape = Sphere(3)
    before = builder.to_mesh(shape)
    policy = VerificationPolicy(0.01, feature_size_mm=0.3)
    precise = verification_mesh(shape, policy)
    after = builder.to_mesh(shape)
    assert len(precise.faces) > len(before.faces)
    assert len(before.faces) == len(after.faces)
    assert precise.extents == pytest.approx([6, 6, 6], abs=0.02)
    assert policy.provenance(0.1)["warnings"] == []
    assert display_provenance(0.1)["absolute_deflection_mm"] is None


def test_small_feature_and_loose_acceptance_budget_are_reported_independently():
    policy = VerificationPolicy(0.1, feature_size_mm=0.3)
    messages = policy.provenance(0.2)["warnings"]
    assert any("acceptance tolerance" in message for message in messages)
    assert any("smallest feature" in message for message in messages)
    assert VerificationPolicy(0.025).provenance()["warnings"]


def test_verification_kills_a_meshing_process_that_exceeds_its_time_budget():
    started = time.monotonic()
    with pytest.raises(MeshingError, match="Verification unknown: end-to-end time budget exceeded"):
        verification_mesh(Box(1, 1, 1), VerificationPolicy(0.01, timeout_s=0.001))
    assert time.monotonic() - started < 5


def test_verification_refuses_an_over_budget_mesh_instead_of_coarsening():
    with pytest.raises(MeshingError, match="triangle budget"):
        verification_mesh(Sphere(3), VerificationPolicy(0.01, max_triangles=12))


@pytest.mark.parametrize("options", [
    {"accuracy_mm": 0}, {"accuracy_mm": float("nan")},
    {"accuracy_mm": .01, "timeout_s": 121},
    {"accuracy_mm": .01, "feature_size_mm": 0},
    {"accuracy_mm": .01, "max_triangles": 10.5},
])
def test_bad_policy_is_rejected_before_any_process_starts(options):
    with pytest.raises(ValueError):
        VerificationPolicy(**options)


def test_cancellation_kills_and_reaps_the_owned_child(monkeypatch):
    import os
    import threading
    from nurb import bounded
    stopped=threading.Event();children=[];fork=bounded.os.fork
    def remember():
        pid=fork()
        if pid:children.append(pid);stopped.set()
        return pid
    monkeypatch.setattr(bounded.os,'fork',remember)
    with pytest.raises(bounded.WorkCancelled,match='cancelled'):
        verification_mesh(Sphere(3),VerificationPolicy(.001),stopped.is_set)
    assert len(children)==1
    with pytest.raises(ChildProcessError):os.waitpid(children[0],os.WNOHANG)
