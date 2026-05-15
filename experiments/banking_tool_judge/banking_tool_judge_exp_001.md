# banking_tool_judge_exp_001_current_accounts

This file documents the prototype case and expected input shape for the banking answer-quality LLM judge.

## Scope Split

The LLM judge YAML evaluates only:

- `answer_grounding`
- `answer_completeness`

The following dimensions are evaluated separately outside the YAML, using deterministic scorers from `ai-data-science/evals`:

- `ToolUsageScorer`
  - includes tool call sequence

- `RoutingCorrectnessScorer`

## Runtime Inputs

The answer-quality judge expects each evaluated turn to provide:

- `case_id`: stable case identifier.
- `turn_id`: stable turn identifier within the case.
- `user_message`: user message for this turn.
- `actual_tool_trace`: structured trace of actual tool calls, arguments, outputs, and order. Tool output is the source of truth for grounding.
- `actual_output`: final answer produced by the agent.
- `required_points`: facts or behaviors the final answer must include.
- `grounding_requirements`: case-specific rules for which parts of the tool output should be treated as authoritative.
- `disallowed_claims`: optional claims or answer behaviors that should be treated as failures.
- `evaluation_focus`: optional short note describing the main answer-quality risk.

The deterministic scorer layer can use additional fields such as:

- `expected_tools`
- `tools_called`
- `expected_tool_sequence`
- `actual_tool_sequence`
- `expected_tool_calls`

## Prototype Case

```yaml
case_id: "banking_tool_eval_001"
scenario: "current_accounts_keep_currencies_separate"
description: "Single-tool account listing where the answer must not collapse CZK and EUR into one total."
tags:
  - "accounts"
  - "single_tool"
  - "multi_currency"
turns:
  - turn_id: 1
    user_message: "What current accounts do I have?"
    expected_tool_calls:
      - tool: "banking_get_accounts"
        parameters:
          account_type: "CURRENT"
        rationale: "A current-account listing is enough; no detail lookup is needed because the user asked for the available current accounts and their balances."
        expected_answer_behavior:
          - "Identify the two returned current accounts by human-readable names and balances."
          - "Use the per-account currency fields, not the helper summary total, when describing the balances."
    expected_tools:
      - "banking_get_accounts"
    expected_tool_sequence:
      - step: 1
        tool: "banking_get_accounts"
        reason: "A current-account listing is enough; no detail lookup is needed."
    required_points:
      - "The answer identifies two current accounts: \"Muj bezny ucet\" in CZK and \"Devizovy ucet\" in EUR."
      - "The balances or disposable amounts stay separated by currency."
    grounding_requirements:
      - "Structured account rows matter more than the helper summary string because the helper summary mixes currencies into one CZK total."
      - "The answer should stay at the level of human-readable account names and balances, not internal IDs."
    disallowed_claims:
      - "Do not report one combined total across CZK and EUR."
      - "Do not expose IBANs or account IDs unless the user asked for them."
    evaluation_focus: "Judge whether the model interpreted the mixed-currency result correctly rather than parroting the helper summary."
```

