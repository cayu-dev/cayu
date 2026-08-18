from examples.cost_quality_comparison import build_demo_report


def test_deterministic_cost_quality_example_demonstrates_every_status() -> None:
    report = build_demo_report().model_dump(mode="json")

    assert [pair["status"] for pair in report["pairs"]] == [
        "measured_unmatched",
        "unavailable",
        "unpriced",
        "verified",
    ]
    assert report["aggregate"]["eligible_pair_ids"] == ["verified"]
    assert [item["pair_id"] for item in report["aggregate"]["exclusions"]] == [
        "measured-unmatched",
        "unavailable",
        "unpriced",
    ]
    serialized = str(report)
    for unsafe_field in ("prompt", "messages", "model_output", "credentials", "metadata"):
        assert unsafe_field not in serialized
