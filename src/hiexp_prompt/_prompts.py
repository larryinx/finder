"""Paper-verbatim prompts used by the offline HEK construction pipeline.

The two prompts come from the HiExp paper (Hao et al., 2026,
*Beyond Stochastic Exploration*):
  - Table 11: contrastive distillation, used to turn (success, failure)
    rollout pairs into atomic E1 entries.
  - Table 12: hierarchical clustering, used to turn a cluster of E1
    entries into one strategy-level E2 entry.

We render task-specific inputs (the question, success/failure trajectory
text, or the cluster body) into these templates by string concatenation;
the templates themselves are kept verbatim so the JSON output schema
matches what ``hiexp.experience.schema.ExperienceEntry`` expects.
"""

from __future__ import annotations

# Paper Table 11 (verbatim).
CONTRASTIVE_DISTILLATION_INSTRUCTION = (
    "An agent system is provided with a set of experiences and has tried to solve "
    "the question multiple times with both successful and wrong solutions. Review "
    "these problem-solving attempt and extract generalizable experiences. Follow "
    "these steps:\n\n"
    "1. Trajectory Analysis:\n"
    "- For successful steps: Identify key correct decisions, insights and formats used\n"
    "- For errors: Pinpoint where and why the reasoning, answer or formatting went wrong\n"
    "- Note any important patterns or strategies usedmissed\n"
    "- Review why some trajectories fail? Is there any key steps are missed, or formats are wrong?\n\n"
    "2. Experiences Summarization:\n"
    "- Summarize and output with the following format: {\n"
    '  "type": "The category to classify the question, including domain and solving method",\n'
    '  "title": "A one-sentence summary of the general experience",\n'
    '  "tags": ["Key words or tags, fewer than 5 words"],\n'
    '  "description": "Your analysis here, within 100 words",\n'
    '  "thinking": "Your thinking process here, especially comparing correct and incorrect solution attempts, within 100 words"\n'
    "}"
)

# Paper Table 12 (verbatim).
HIERARCHICAL_CLUSTERING_INSTRUCTION = (
    "You are given a set of experiences that an agent has accumulated while solving "
    "various questions. Your task is to cluster the similar experiences into "
    "generalized experiences that capture the core patterns and strategies. These "
    "generalized experiences should enable the agent to solve similar questions "
    "correctly and efficiently in the future.\n"
    "The set of experiences is listing with the following format: "
    '[{"type": "", "title": "", "tags": "", "description": "", "thinking": "", '
    '"qa_groups": ["id":"", ...]}, ...], where qa_groups is the questions and '
    "answers in this experience group.\n\n"
    "Summarize and output with the following format: "
    '[{ "ids": ["all qa ids from the experiences in this cluster"], '
    '"type": "A category for this group of questions, including domain and solving method.", '
    '"title": "A one-sentence summary of the generalized strategy for this cluster.", '
    '"tags": ["A list of up to 5 keywords or tags."], '
    '"description": "Your analysis of the common patterns and core logic for this cluster, within 100 words.", '
    '"thinking": "Your thinking process here, especially differences within the group of experiences, within 100 words" }]'
)


def build_contrastive_distillation_prompt(
    question: str,
    successful_trajectory_text: str,
    failed_trajectory_text: str,
) -> str:
    """Wrap Table 11 around one (question, +traj, -traj) triple."""
    return (
        f"{CONTRASTIVE_DISTILLATION_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        f"Successful trajectory:\n{successful_trajectory_text}\n\n"
        f"Failed trajectory:\n{failed_trajectory_text}\n\n"
        "Output ONLY the JSON object described above; no surrounding prose."
    )


def build_clustering_prompt(cluster_entries_json: str) -> str:
    """Wrap Table 12 around a JSON list of E1 entries."""
    return (
        f"{HIERARCHICAL_CLUSTERING_INSTRUCTION}\n\n"
        f"Input experiences:\n{cluster_entries_json}\n\n"
        "Output ONLY the JSON array described above; no surrounding prose."
    )
