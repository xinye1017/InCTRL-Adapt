# Orchestrator Skills

## 1. Task Breakdown Protocol
When receiving a high-level goal:
1. Identify the core components: Data, Model, Training, Evaluation, Research.
2. Draft a sequence of operations (e.g., 1. Research -> 2. Data -> 3. Model -> 4. Train -> 5. Eval).
3. Validate that no single task overlaps multiple agent domains.

## 2. Delegation Protocol
For each sub-task:
1. Identify the target agent using `.claude/ROUTER.md`.
2. Formulate a specific, context-rich prompt for that agent.
3. Define the exact expected output format from that agent (e.g., "Provide a JSON config", "Provide a modified PyTorch class").

## 3. Aggregation Protocol
When agents return results:
1. Verify all constraints and requirements were met.
2. Resolve conflicting decisions between agents (e.g., Data Engineer wants batch size 256, Training Engineer says GPU max is 64 -> enforce 64).
3. Present the final consolidated pipeline to the user.
