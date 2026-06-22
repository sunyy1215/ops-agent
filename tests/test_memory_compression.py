from ops_rag_agent.memory.compression import (
    recompress_with_reflection,
    reflect_compression_summary,
)


def test_reflection_detects_missing_tool_results() -> None:
    middle = [
        {"role": "tool", "content": "kubectl describe pod api-0 returned CrashLoopBackOff"},
        {"role": "assistant", "content": "Need to verify restart count and image pull status."},
    ]
    bad_summary = (
        "task_goal: continue current workflow\n"
        "confirmed_facts: ['pod unhealthy']\n"
        "actions_taken: []\n"
        "pending_questions: []\n"
        "constraints: ['preserve head and tail zones']\n"
        "next_best_actions: continue from latest protected tail"
    )

    reflection = reflect_compression_summary(bad_summary, middle)

    assert reflection["passed"] is False
    assert any("tool results" in item for item in reflection["feedback"])


def test_recompress_with_reflection_includes_feedback_context() -> None:
    middle = [
        {"role": "tool", "content": "prometheus returned cpu_usage=92% on node-a"},
        {"role": "assistant", "content": "Need to keep the spike evidence in the summary."},
    ]

    summary = recompress_with_reflection(
        middle,
        ["summary should retain important tool results"],
        retry_count=1,
    )

    assert "reflection_feedback=" in summary
    assert "cpu_usage=92%" in summary
