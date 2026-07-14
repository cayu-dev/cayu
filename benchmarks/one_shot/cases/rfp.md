# RFP management workflow

## Initial prompt

Build me an agent to automate the workflow of managing RFPs

## Predetermined clarification answers

- One local proposal manager uses it.
- An RFP arrives as a structured record with title, issuer, due date, requested
  capabilities, evaluation criteria, and source-document artifact IDs.
- The model summarizes requirements, identifies missing information, and drafts
  a bid/no-bid recommendation.
- Persist the RFP state and recommendation durably.
- Preparing a response plan is safe; marking bid/no-bid is an external business
  decision and must pause for human approval.
- No email, submission portal, or live provider call is required for acceptance.
- Recovery must not duplicate an approved decision after interruption.

## Case-specific acceptance

The vertical slice includes a domain input, model decision, explicit-effect
decision tool, enforcing approval boundary, durable state, deterministic runtime
test, and trajectory eval that observes the approval interruption.
