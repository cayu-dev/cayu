/**
 * A made-up finance app with three specialists. It exercises every state the
 * dashboard shows: approvals, questions, recovery, knowledge, subagents, a browser,
 * failures, sessions without an app-set title, and labels that vary.
 */
import {
  buildSession,
  knowledgeItem,
  type SessionSpec,
  seedTask,
  specialist,
  tool,
} from "../builder"
import type { AppSeed } from "../model"

const NOW = Date.parse("2026-10-21T23:30:00Z")
const CLOSE_GRAPH = "graph-close-2026-10"
const ctx = { now: NOW, runtime: "0.13.0", provider: "openai", model: "gpt-6-luna" }

const specialists = [
  specialist(ctx, "close-agent", "Month-end close: accruals, reconciliations and journal entries", [
    tool("query_ledger", "Query general ledger lines.", { verb: "Queried the ledger" }, "none", {
      source: { kind: "package", package: "ledger-connect", version: "3.2.0" },
      schema: {
        type: "object",
        properties: {
          account: { type: "string" },
          period: { type: "string" },
          cost_centers: { type: "array", items: { type: "string" } },
        },
      },
    }),
    tool("match_purchase_orders", "Match ledger lines to open purchase orders."),
    tool("fetch_vendor", "Look up a vendor record.", { verb: "Looked up", noun: "vendor" }),
    tool(
      "post_journal_entry",
      "Post a journal entry to the ledger.",
      { verb: "Posted", noun: "journal entry", plural: "journal entries" },
      "external",
      {
        approval: "always",
        schema: {
          type: "object",
          properties: { entry: { type: "string" }, lines: { type: "integer" } },
          required: ["entry"],
        },
      },
    ),
  ]),
  specialist(ctx, "anomaly-agent", "Investigates unusual movements in the general ledger", [
    tool("compare_periods", "Compare an account across two periods.", { verb: "Compared periods" }),
    tool("search_contracts", "Search vendor contracts.", { verb: "Searched contracts" }, "none", {
      source: { kind: "mcp", server: "contracts" },
    }),
    tool("browser_session", "Use a browser in a sandbox."),
    tool("subagent", "Start a subagent session with another specialist."),
  ]),
  specialist(ctx, "ap-auditor", "Audits accounts-payable invoice batches before payment", [
    tool("load_invoice_batch", "Load an invoice batch.", {
      verb: "Loaded",
      noun: "invoice batch",
      plural: "invoice batches",
    }),
    tool("find_duplicates", "Find likely duplicate invoices.", { verb: "Checked for duplicates" }),
    tool(
      "hold_payment",
      "Hold an invoice out of the payment run.",
      { verb: "Held", noun: "payment" },
      "external",
      {
        approval: "sometimes",
        schema: {
          type: "object",
          properties: {
            invoice: { type: "string" },
            vendor: { type: "string" },
            amount: { type: "string" },
            reason: { type: "string" },
          },
          required: ["invoice", "reason"],
        },
      },
    ),
  ]),
]

const specs: SessionSpec[] = [
  {
    id: "close-00a4224687bc",
    agent: "close-agent",
    status: "running",
    startedAgo: 3,
    title: "Intercompany elimination schedule, October",
    labels: { period: "2026-10", entity: "US01" },
    steps: [
      { user: "Prepare the October intercompany elimination schedule for US01, UK02 and DE03." },
      { think: "Pulling intercompany pairs first" },
      {
        tool: "query_ledger",
        args: {
          account: "1900 Intercompany",
          entities: ["US01", "UK02", "DE03"],
          period: "2026-10",
        },
        result: "36 intercompany pairs, net difference €2,310.40",
      },
      {
        say: "Pulled 36 intercompany pairs. UK02 ↔ DE03 is off by €2,310.40; reconciling that pair now.",
      },
      {
        pending: "query_ledger",
        args: { account: "1900 Intercompany", entities: ["UK02", "DE03"], detail: true },
      },
    ],
  },
  {
    id: "ap-ai-c33e300e024d3a41afd6",
    agent: "ap-auditor",
    status: "running",
    startedAgo: 22,
    title: "Invoice batch AP-2026-10-21",
    labels: { batch: "AP-2026-10-21" },
    waiting: {
      kind: "approval",
      tool: "hold_payment",
      reason: "Holding a payment over $10,000 needs a person's approval.",
    },
    steps: [
      {
        user: "Audit invoice batch AP-2026-10-21 (42 invoices) for duplicates and hold anything suspicious before the 5pm payment run.",
      },
      {
        tool: "load_invoice_batch",
        args: { batch: "AP-2026-10-21" },
        result: "42 invoices, $388,120.55",
      },
      {
        tool: "find_duplicates",
        args: { strategy: "vendor+amount+date±7d" },
        result: "1 likely duplicate: INV-88213 matches INV-88197 (paid Oct 14)",
      },
      {
        say: "INV-88213 from Acme Office Supply ($12,940.00) matches INV-88197, paid on Oct 14: same amount, same PO line, invoice dates three days apart. Holding it keeps it out of today's payment run.",
      },
      {
        pending: "hold_payment",
        args: {
          invoice: "INV-88213",
          vendor: "Acme Office Supply",
          amount: "12,940.00 USD",
          reason: "Likely duplicate of INV-88197",
        },
      },
    ],
  },
  {
    id: "close-99131a7ace99",
    agent: "close-agent",
    status: "running",
    startedAgo: 300,
    title: "October accruals, procurement cost centers",
    labels: { period: "2026-10", cost_center: "CC-4300" },
    waiting: {
      kind: "question",
      question:
        "Two Northwind Freight accruals ($18,420.00) have no PO, and the vendor's default cost center is outside the scope you gave me. Where should they post?",
      options: [
        "CC-4100 Logistics (vendor default)",
        "CC-4300 Procurement ops",
        "Leave unposted and flag",
      ],
    },
    steps: [
      {
        user: "Reconcile October accruals for the procurement cost centers (CC-4300, CC-4310). Flag anything without a PO.",
      },
      {
        tool: "query_ledger",
        args: {
          account: "2100 Accrued liabilities",
          cost_centers: ["CC-4300", "CC-4310"],
          period: "2026-10",
        },
        result: "214 lines, $1,284,906.12",
      },
      { tool: "match_purchase_orders", args: { lines: 214 }, result: "209 matched, 5 unmatched" },
      {
        tool: "fetch_vendor",
        args: { vendor: "Northwind Freight" },
        result: "Vendor 10442, default cost center CC-4100",
      },
      {
        say: "I matched 209 of 214 lines to open POs. Three of the five unmatched lines are timing differences that reverse on Nov 2, noted in the workpaper. The other two are Northwind Freight lines with no PO.",
      },
    ],
  },
  {
    id: "close-e26d850c6f4e",
    agent: "close-agent",
    status: "interrupted",
    startedAgo: 145,
    labels: { period: "2026-10" },
    waiting: {
      kind: "recovery",
      tool: "post_journal_entry",
      detail: "Tool started but no terminal result was recorded.",
    },
    steps: [
      {
        user: "Post reclass entries for the software subscriptions miscoded to 6400 Office supplies.",
      },
      {
        tool: "query_ledger",
        args: { account: "6400 Office supplies", vendor_category: "software", period: "2026-10" },
        result: "11 lines, $23,551.08",
      },
      { pending: "post_journal_entry", args: { entry: "JE-DRAFT-0193", lines: 22 } },
    ],
  },
  {
    id: "anomaly-case-423b23a3659dfa336b81",
    agent: "anomaly-agent",
    status: "completed",
    startedAgo: 60 * 26,
    seconds: 412,
    title: "Freight expense up 38% in week 41",
    summary: "Explained: fuel surcharge",
    labels: { account: "6100", week: "41" },
    steps: [
      { user: "Why did freight expense jump 38% in week 41?" },
      {
        tool: "compare_periods",
        args: { account: "6100 Freight", a: "W40", b: "W41" },
        result: "+$61,212.80 (+38.4%)",
      },
      {
        tool: "subagent",
        args: { specialist: "close-agent", task: "Pull GL detail for 6100 Freight, weeks 39–41" },
        result: "Session close-a1f0c27d9e31 finished: 1,102 lines exported",
      },
      {
        tool: "browser_session",
        args: { url: "https://carrier.example.com/rates" },
        result: "Rate page returned 503",
        error: true,
      },
      {
        tool: "search_contracts",
        args: { query: "Northwind fuel surcharge" },
        result: "Contract amendment #3, effective Oct 6: 14% fuel surcharge on all lanes",
      },
      {
        say: "Most of the jump ($48,900 of $61,213) comes from Northwind Freight. Contract amendment #3 added a 14% fuel surcharge from October 6, which first reached invoices in week 41. The rest is two expedited shipments for the DE03 plant. The carrier's rate page was down, so the surcharge figure comes from the contract, not the carrier's system.",
      },
    ],
  },
  {
    id: "close-a1f0c27d9e31",
    agent: "close-agent",
    status: "completed",
    startedAgo: 60 * 26 - 2,
    seconds: 95,
    parent: "anomaly-case-423b23a3659dfa336b81",
    labels: { account: "6100" },
    steps: [
      { user: "Pull GL detail for 6100 Freight, weeks 39–41." },
      {
        tool: "query_ledger",
        args: { account: "6100 Freight", weeks: [39, 40, 41] },
        result: "1,102 lines",
      },
      { say: "Exported 1,102 lines for weeks 39–41." },
    ],
  },
  {
    id: "anomaly-case-286773a04e01bebc35ec",
    agent: "anomaly-agent",
    status: "completed",
    startedAgo: 60 * 30,
    seconds: 268,
    title: "Negative balance in 2150 Accrued payroll",
    summary: "Duplicate posting found",
    summaryTone: "attention",
    labels: { account: "2150" },
    steps: [
      { user: "Explain the negative balance in 2150 Accrued payroll." },
      {
        tool: "compare_periods",
        args: { account: "2150 Accrued payroll", a: "2026-09", b: "2026-10" },
        result: "Balance −$42,118.00 (was +$17,940.00)",
      },
      {
        tool: "browser_session",
        args: { url: "https://payroll.example.com/runs/2026-10-15" },
        result: "Payroll run 2026-10-15: gross $1.92M, posted by integration Oct 15",
      },
      {
        output: {
          cause: "Payroll run 2026-10-15 posted twice",
          entries: ["Integration post, Oct 15", "JE-55120, manual, Oct 16"],
          fix: "Reverse JE-55120",
          balance_after_fix: "+$17,940.00",
        },
      },
    ],
  },
  {
    id: "close-537587ecae8f",
    agent: "close-agent",
    status: "failed",
    startedAgo: 60 * 38,
    seconds: 64,
    reason: "Model provider rate limit (429) after 3 retries. Nothing was posted.",
    labels: { period: "2026-10" },
    steps: [
      { user: "Pull Oct 31 FX rates and revalue EUR and GBP payables." },
      {
        tool: "query_ledger",
        args: { account: "2000 Accounts payable", currencies: ["EUR", "GBP"] },
        result: "418 open items",
      },
    ],
  },
  {
    id: "close-96f54e408b29",
    agent: "close-agent",
    status: "interrupted",
    startedAgo: 60 * 50,
    seconds: 130,
    title: "October flux commentary for the CFO pack",
    labels: { period: "2026-10" },
    steps: [
      { user: "Draft the October flux commentary for the CFO pack." },
      {
        tool: "compare_periods",
        args: { scope: "P&L", a: "2026-09", b: "2026-10" },
        result: "14 lines over the 10% threshold",
      },
      { say: "Drafting commentary for the 14 lines over threshold, starting with revenue." },
    ],
  },
  {
    id: "close-02424b48a511",
    agent: "close-agent",
    status: "completed",
    startedAgo: 60 * 30 + 15,
    seconds: 188,
    title: "Prepaid insurance roll-forward",
    summary: "Schedule updated, 1 adjustment",
    labels: { period: "2026-10", account: "1400" },
    steps: [
      { user: "Roll the prepaid insurance schedule forward to October." },
      {
        tool: "query_ledger",
        args: { account: "1400 Prepaid insurance", period: "2026-10" },
        result: "6 policies, $214,800.00",
      },
      {
        say: "Rolled forward six policies. Policy PL-2291 amortizes over 11 months, not 12, because coverage started Nov 15, so I adjusted October's expense by $1,240.00.",
      },
    ],
  },
  {
    id: "ap-ai-7d21f09b44e1c2a0",
    agent: "ap-auditor",
    status: "completed",
    startedAgo: 60 * 75,
    seconds: 342,
    title: "W-9 check for October's new suppliers",
    summary: "1 missing W-9",
    summaryTone: "attention",
    labels: { batch: "vendors-2026-10" },
    steps: [
      { user: "Check W-9s for the six suppliers added in October." },
      { tool: "load_invoice_batch", args: { vendors_added: "2026-10" }, result: "6 suppliers" },
      {
        output: {
          checked: 6,
          missing: [{ vendor: "Brightline Staffing", vendor_id: 10511 }],
          expired: [],
        },
      },
    ],
  },
  // Eval trials: sessions the eval runner started (runtime-owned `cayu:eval_trial`).
  // Counts and Needs attention leave them out; the Sessions list shows them on request.
  {
    id: "close-eval-7f3a91c2d4e0",
    agent: "close-agent",
    status: "completed",
    startedAgo: 95,
    seconds: 41,
    metadata: { "cayu:eval_trial": true },
    steps: [
      { user: "Reconcile accruals for CC-4300, period 2026-09, and flag lines without a PO." },
      {
        tool: "query_ledger",
        args: { account: "2100 Accrued liabilities", period: "2026-09", cost_centers: ["CC-4300"] },
        result: "58 lines, $212,904.10",
      },
      {
        tool: "match_purchase_orders",
        args: { lines: 58 },
        result: "56 matched, 2 without a PO",
      },
      {
        say: "Reconciled 58 accrual lines for CC-4300. Two have no PO: Northwind Freight ($6,120.00) and Lakeside Print ($940.00).",
      },
    ],
  },
  {
    id: "ap-eval-2b81c0e94f57",
    agent: "ap-auditor",
    status: "failed",
    startedAgo: 93,
    seconds: 18,
    reason: "Eval case limit reached: 4 tool calls allowed, a 5th was attempted.",
    metadata: { "cayu:eval_trial": true },
    steps: [
      { user: "Audit invoice batch AP-EVAL-007 (12 invoices) for duplicates." },
      {
        tool: "load_invoice_batch",
        args: { batch: "AP-EVAL-007" },
        result: "12 invoices, $48,210.00",
      },
      {
        tool: "find_duplicates",
        args: { strategy: "vendor+amount+date±7d" },
        result: "No duplicates",
      },
    ],
  },
  {
    id: "close-c6325f3696f6",
    agent: "close-agent",
    status: "completed",
    startedAgo: 60 * 24 * 12,
    seconds: 501,
    title: "Bank reconciliation, account 1010",
    summary: "Reconciled",
    labels: { period: "2026-09", account: "1010" },
    steps: [
      { user: "Tie out the September bank reconciliation for account 1010 Operating." },
      {
        tool: "query_ledger",
        args: { account: "1010 Operating", period: "2026-09" },
        result: "1,904 lines",
      },
      { say: "Reconciled. Two outstanding checks (#20417, #20431) clear in October." },
    ],
  },
]

const built = specs.map((s) => buildSession(ctx, s))
const sessions = built.map((b) => b.seed)
const child = sessions.find((s) => s.session.id === "anomaly-case-423b23a3659dfa336b81")
if (child) child.overview.uses.subagents = 1

export const monthEnd: AppSeed = {
  id: "month-end",
  label: "Month-end close",
  description: "Made-up finance agent with three specialists and every state",
  now: new Date(NOW).toISOString(),
  app: { name: "finance-monthend", runtime_version: ctx.runtime },
  specialists: { agents: specialists, total_count: specialists.length },
  sessions,
  inbox: [
    ...built.flatMap((b) => (b.inbox ? [b.inbox] : [])),
    knowledgeItem(
      "k-pl2291",
      "close-agent",
      "close-02424b48a511",
      "Policy PL-2291 amortizes over 11 months because coverage started Nov 15.",
      NOW - 60_000 * 60 * 29,
    ),
    knowledgeItem(
      "k-northwind",
      "anomaly-agent",
      "anomaly-case-423b23a3659dfa336b81",
      "Northwind Freight's 14% fuel surcharge applies to all lanes from 2026-10-06 (contract amendment #3).",
      NOW - 60_000 * 60 * 25,
    ),
  ],
  taskTypes: [
    {
      name: "reconcile_accruals",
      description:
        "Reconcile one period's accruals for a set of cost centers and flag lines without a PO.",
      agent_name: "close-agent",
      input_schema: {
        type: "object",
        properties: {
          period: { type: "string", description: "Accounting period, e.g. 2026-10" },
          cost_centers: {
            type: "array",
            items: { type: "string" },
            description: "Cost centers, comma separated",
          },
        },
        required: ["period"],
      },
    },
    {
      name: "audit_invoice_batch",
      description: "Audit an AP invoice batch for duplicates before the payment run.",
      agent_name: "ap-auditor",
      input_schema: {
        type: "object",
        properties: { batch: { type: "string", description: "Batch ID, e.g. AP-2026-10-21" } },
        required: ["batch"],
      },
    },
    {
      name: "investigate_anomaly",
      description: "Explain an unusual movement in a general-ledger account.",
      agent_name: "anomaly-agent",
      input_schema: {
        type: "object",
        properties: {
          account: { type: "string", description: "Account, e.g. 6100" },
          question: { type: "string", description: "What looks wrong" },
        },
        required: ["account", "question"],
      },
    },
  ],
  knowledge: [
    {
      id: "k-fx-source",
      text: "Month-end FX revaluation uses the ECB reference rate published on the last business day, not the bank's rate.",
      kind: "fact",
      agent_name: "close-agent",
      source_session_id: "close-537587ecae8f",
      kept_by: "maria.chen",
      kept_at: new Date(NOW - 864e5 * 9).toISOString(),
      lookups: 14,
      last_used_at: new Date(NOW - 60_000 * 60 * 38).toISOString(),
    },
    {
      id: "k-northwind-cc",
      text: "Northwind Freight accruals post to CC-4100 Logistics unless the PO says otherwise.",
      kind: "rule",
      agent_name: "close-agent",
      source_session_id: null,
      kept_by: "maria.chen",
      kept_at: new Date(NOW - 864e5 * 20).toISOString(),
      lookups: 31,
      last_used_at: new Date(NOW - 60_000 * 60 * 5).toISOString(),
    },
    {
      id: "k-dup-window",
      text: "Treat invoices from the same vendor, same amount and same PO line within 7 days as likely duplicates.",
      kind: "rule",
      agent_name: "ap-auditor",
      source_session_id: "ap-ai-7d21f09b44e1c2a0",
      kept_by: "j.okafor",
      kept_at: new Date(NOW - 864e5 * 6).toISOString(),
      lookups: 8,
      last_used_at: new Date(NOW - 60_000 * 22).toISOString(),
    },
  ],
  // A close graph (one prerequisite on its second attempt), a scheduled reversal, a held
  // task, a subagent's dispatch task, and two of Cayu's own housekeeping tasks.
  tasks: [
    seedTask(NOW, {
      id: "task-close-books",
      title: "Close October books",
      type: "close_books",
      status: "waiting_dependencies",
      agent: "close-agent",
      createdAgo: 60 * 40,
      updatedAgo: 4,
      graph: CLOSE_GRAPH,
      prerequisites: [
        "task-ic-elim",
        "task-accruals-procurement",
        "task-prepaid",
        "task-fx-rates-2",
      ],
    }),
    seedTask(NOW, {
      id: "task-ic-elim",
      title: "Intercompany elimination schedule, October",
      type: "reconcile_accruals",
      status: "running",
      agent: "close-agent",
      createdAgo: 4,
      updatedAgo: 0,
      session: "close-00a4224687bc",
      graph: CLOSE_GRAPH,
    }),
    seedTask(NOW, {
      id: "task-accruals-procurement",
      title: "October accruals, procurement cost centers",
      type: "reconcile_accruals",
      status: "running",
      agent: "close-agent",
      createdAgo: 301,
      updatedAgo: 2,
      session: "close-99131a7ace99",
      graph: CLOSE_GRAPH,
    }),
    seedTask(NOW, {
      id: "task-prepaid",
      title: "Prepaid insurance roll-forward",
      type: "reconcile_accruals",
      status: "completed",
      agent: "close-agent",
      createdAgo: 60 * 30 + 16,
      updatedAgo: 60 * 30 + 12,
      session: "close-02424b48a511",
      graph: CLOSE_GRAPH,
    }),
    seedTask(NOW, {
      id: "task-fx-rates-1",
      title: "Revalue EUR and GBP payables at Oct 31 rates",
      type: "revalue_fx",
      status: "failed",
      agent: "close-agent",
      createdAgo: 60 * 38 + 1,
      updatedAgo: 60 * 38 - 1,
      session: "close-537587ecae8f",
      reason: "Model provider rate limit (429) after 3 retries. Nothing was posted.",
      retry: { attempt: 1, max: 3, disposition: "retry_scheduled", successor: "task-fx-rates-2" },
      graph: CLOSE_GRAPH,
    }),
    seedTask(NOW, {
      id: "task-fx-rates-2",
      title: "Revalue EUR and GBP payables at Oct 31 rates",
      type: "revalue_fx",
      status: "pending",
      agent: "close-agent",
      createdAgo: 60 * 38 - 1,
      updatedAgo: 10,
      availableIn: 6,
      retry: {
        attempt: 2,
        max: 3,
        disposition: "active",
        predecessor: "task-fx-rates-1",
        nextIn: 6,
      },
      graph: CLOSE_GRAPH,
    }),
    seedTask(NOW, {
      id: "task-accrual-reversal",
      title: "Reverse October accruals",
      type: "reverse_accruals",
      status: "pending",
      agent: "close-agent",
      createdAgo: 60 * 20,
      updatedAgo: 60 * 3,
      // Nov 1, 6:00 AM Pacific (PST from that morning).
      availableIn: (Date.parse("2026-11-01T14:00:00Z") - NOW) / 60_000,
      scheduled: { revision: 2 },
    }),
    seedTask(NOW, {
      id: "task-vendor-sync",
      title: "Sync vendor master from procurement",
      type: "sync_vendor_master",
      status: "blocked",
      agent: "ap-auditor",
      createdAgo: 60 * 6,
      updatedAgo: 60 * 5,
      reason: "Procurement hasn't sent the October vendor file.",
    }),
    seedTask(NOW, {
      id: "task-ap-batch",
      title: "Invoice batch AP-2026-10-21",
      type: "audit_invoice_batch",
      status: "running",
      agent: "ap-auditor",
      createdAgo: 23,
      updatedAgo: 1,
      session: "ap-ai-c33e300e024d3a41afd6",
    }),
    seedTask(NOW, {
      id: "task-anomaly-freight",
      title: "Freight expense up 38% in week 41",
      type: "investigate_anomaly",
      status: "completed",
      agent: "anomaly-agent",
      createdAgo: 60 * 26 + 1,
      updatedAgo: 60 * 26 - 7,
      session: "anomaly-case-423b23a3659dfa336b81",
    }),
    seedTask(NOW, {
      id: "task-gl-detail",
      title: "Pull GL detail for 6100 Freight, weeks 39–41",
      type: "cayu.dispatch",
      status: "completed",
      agent: "close-agent",
      createdAgo: 60 * 26 - 2,
      updatedAgo: 60 * 26 - 4,
      session: "close-a1f0c27d9e31",
      parent: "task-anomaly-freight",
    }),
    seedTask(NOW, {
      id: "task-knowledge-fx",
      title: "Enrich knowledge: FX rate source",
      type: "cayu.knowledge-enrichment.v1",
      status: "completed",
      agent: null,
      createdAgo: 60 * 37,
      updatedAgo: 60 * 37 - 1,
    }),
    seedTask(NOW, {
      id: "task-ap-deadline",
      title: "Approval deadline, invoice batch AP-2026-10-21",
      type: "cayu.external-wait-deadline.v1",
      status: "pending",
      agent: null,
      createdAgo: 20,
      availableIn: 60 * 4,
      scheduled: {},
    }),
  ],
  taskInputs: {
    "task-close-books": { period: "2026-10", entities: ["US01", "UK02", "DE03"] },
    "task-fx-rates-1": { period: "2026-10", currencies: ["EUR", "GBP"], rate_date: "2026-10-31" },
    "task-fx-rates-2": { period: "2026-10", currencies: ["EUR", "GBP"], rate_date: "2026-10-31" },
    "task-accrual-reversal": { period: "2026-10", reverse_on: "2026-11-01" },
    "task-vendor-sync": { source: "procurement", period: "2026-10" },
    "task-knowledge-fx": { entry_id: "k-fx-source", source_session_id: "close-537587ecae8f" },
    "task-ap-deadline": {
      wait: "approval",
      session_id: "ap-ai-c33e300e024d3a41afd6",
      on_expiry: "escalate",
    },
  },
}
