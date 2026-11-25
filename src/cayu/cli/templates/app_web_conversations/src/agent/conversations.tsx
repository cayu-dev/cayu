import * as React from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";
import { queryOptions, useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import { AppBackLink, AppDetailHeader, AppDetailPage, AppPageHeader, AppPageTitle } from "@kit/layouts/app";
import { AsyncContent, DataTable } from "@kit/patterns";
import { Alert, AlertDescription, Button, Card, CardContent, Field, FieldLabel, Input, Skeleton, StatusBadge, Textarea } from "@kit/ui";
import { formatDateTime, formatRelative } from "@kit/lib";
import { isApiError, queryContentProps, queryTableProps, type Requester, watchInterval } from "@kit/data";
import { BubbleChatIcon } from "@kit/icons";
import { z } from "zod";
import type { AppArea } from "../app/area";
import { useRequester, useWhoami } from "../app/services";
import { conversations, messageOf, ReplyFailed, stream, textOf, titleOf } from "./cayu";
import { Markdown } from "./markdown";

/**
 * THE AGENT'S OWN. Conversations with the agent: the ones it has had, a new one, and one
 * conversation to read and continue. They go through Cayu's API (./cayu.ts), so a role that
 * may not change things reads them and doesn't start or continue one.
 */
export const conversationsArea: AppArea = {
  id: "conversations",
  label: "Conversations",
  icon: BubbleChatIcon,
  path: "/conversations",
  routes: () => [
    { path: "conversations", Component: ConversationsPage },
    { path: "conversations/new", Component: NewConversationPage },
    { path: "conversations/:id", Component: ConversationPage },
  ],
};

export const conversationPath = (id: string) => `/conversations/${encodeURIComponent(id)}`;

/** The start of every conversation list's query key: invalidating it refreshes them all. */
export const conversationsKey = ["conversations"] as const;

/** `GET /app/agent` (integrations/http/__init__.py): the agent the app's conversations are with. */
const agentQuery = (request: Requester) =>
  queryOptions({
    queryKey: ["agent"],
    queryFn: ({ signal }) => request("agent", { signal, schema: z.object({ name: z.string() }) }),
    staleTime: Number.POSITIVE_INFINITY,
  });

function useAgent() {
  return useQuery(agentQuery(useRequester())).data?.name;
}

const STATUS_TONE: Record<string, "neutral" | "progress" | "success" | "warning" | "danger"> = {
  running: "progress",
  completed: "success",
  failed: "danger",
  paused: "warning",
  interrupted: "warning",
};

function ConversationsPage() {
  const agent = useAgent();
  const { mayChange } = useWhoami();
  // Pages of 50, the next one asked for only when the person wants more.
  const pages = useInfiniteQuery({
    queryKey: [...conversationsKey, agent, "all"],
    queryFn: ({ pageParam, signal }) => conversations.list(agent ?? "", pageParam, 50, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor,
    enabled: agent !== undefined,
  });
  const rows = pages.data?.pages.flatMap((page) => page.rows) ?? [];
  return (
    <div className="flex flex-col gap-page-section" data-origin="own">
      <AppPageHeader
        description={agent ? `Conversations with ${agent}, newest activity first.` : null}
        actions={mayChange ? <Button render={<Link to="/conversations/new" />}>New conversation</Button> : null}
      >
        <AppPageTitle>Conversations</AppPageTitle>
      </AppPageHeader>
      <DataTable
        {...queryTableProps(pages, "Couldn't load the conversations.")}
        label="Conversations"
        showHeading={false}
        rows={rows}
        rowId={(row) => row.id}
        searchText={(row) => `${titleOf(row)} ${row.id}`}
        itemLabel={{ singular: "conversation", plural: "conversations" }}
        emptyTitle="No conversations yet"
        emptyDescription={mayChange ? "Start one to ask the agent something." : "None yet."}
        columns={[
          {
            key: "title",
            label: "Conversation",
            sortValue: (row) => titleOf(row),
            cell: (row) => (
              <Link to={conversationPath(row.id)} className="font-medium underline-offset-4 hover:underline">
                {titleOf(row)}
              </Link>
            ),
          },
          {
            key: "status",
            label: "Status",
            sortValue: (row) => row.status,
            cell: (row) => <StatusBadge tone={STATUS_TONE[row.status] ?? "neutral"}>{row.status}</StatusBadge>,
          },
          {
            key: "activity",
            label: "Last activity",
            sortValue: (row) => row.updated_at,
            cell: (row) => (
              <time dateTime={row.updated_at} title={formatDateTime(row.updated_at)}>
                {formatRelative(row.updated_at)}
              </time>
            ),
          },
        ]}
      />
      {pages.hasNextPage ? (
        <div className="flex justify-center">
          <Button variant="outline" disabled={pages.isFetchingNextPage} onClick={() => pages.fetchNextPage()}>
            {pages.isFetchingNextPage ? "Loading…" : "Show more"}
          </Button>
        </div>
      ) : null}
    </div>
  );
}

function NewConversationPage() {
  const agent = useAgent();
  const { mayChange, me } = useWhoami();
  return (
    <AppDetailPage data-origin="own">
      <AppDetailHeader
        back={<AppBackLink render={<Link to="/conversations" />}>Back to conversations</AppBackLink>}
        title="New conversation"
        description={agent ? `With ${agent}.` : null}
      />
      {me && !mayChange ? (
        <ErrorNote message="Your role reads conversations; an admin or a developer starts them." />
      ) : agent ? (
        <NewConversationForm agent={agent} />
      ) : (
        <Skeleton className="h-64 w-full" />
      )}
    </AppDetailPage>
  );
}

function NewConversationForm({ agent }: { agent: string }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [title, setTitle] = React.useState("");
  const [text, setText] = React.useState("");
  const [reply, setReply] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  // Leaving the page keeps reading the reply, since stopping would stop the agent, but it no
  // longer updates this page or opens the conversation.
  const shown = React.useRef(true);
  React.useEffect(() => {
    shown.current = true;
    return () => {
      shown.current = false;
    };
  }, []);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const id = await stream("run", { agent, prompt: text, title: title.trim() || undefined }, (next) => shown.current && setReply(next));
      // The lists don't have it yet, whether or not this page is still shown.
      void queryClient.invalidateQueries({ queryKey: conversationsKey });
      if (shown.current) void navigate(conversationPath(id));
    } catch (caught) {
      // Once it has started, the conversation exists: open it, with what went wrong, rather
      // than start another.
      if (caught instanceof ReplyFailed && caught.sessionId) {
        void queryClient.invalidateQueries({ queryKey: conversationsKey });
        if (shown.current) void navigate(conversationPath(caught.sessionId), { state: { error: messageOf(caught) } });
        return;
      }
      if (!shown.current) return;
      setError(messageOf(caught));
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardContent className="flex flex-col gap-4">
        <Field>
          <FieldLabel>Title (optional)</FieldLabel>
          <Input value={title} placeholder="What it's about" onChange={(event) => setTitle(event.target.value)} />
        </Field>
        <Field>
          <FieldLabel>Message</FieldLabel>
          <Textarea value={text} placeholder="What should the agent do?" onChange={(event) => setText(event.target.value)} />
        </Field>
        <div>
          <Button disabled={busy || !text.trim()} onClick={start}>
            {busy ? "Working…" : "Start"}
          </Button>
        </div>
        {reply ? (
          <div className="rounded-lg bg-muted p-3">
            <Markdown text={reply} />
          </div>
        ) : null}
        {error ? <ErrorNote message={error} /> : null}
      </CardContent>
    </Card>
  );
}

// The conversations whose reply this page is streaming in, followed live until it ends.
// Module-wide, so leaving the page and coming back while one streams still knows.
const streaming = new Set<string>();
const streamingListeners = new Set<() => void>();
function setStreaming(id: string, on: boolean) {
  if (on) streaming.add(id);
  else streaming.delete(id);
  streamingListeners.forEach((listener) => listener());
}
function subscribeStreaming(listener: () => void) {
  streamingListeners.add(listener);
  return () => {
    streamingListeners.delete(listener);
  };
}
function useStreaming(id: string) {
  return React.useSyncExternalStore(subscribeStreaming, () => streaming.has(id));
}

// Statuses that can change without this page: one running elsewhere, or waiting for a person
// to answer or approve in Cayu's inbox, after which the agent carries on.
const UNSETTLED = new Set(["pending", "running", "interrupting", "interrupted", "paused"]);
const watchUnsettled = watchInterval<Awaited<ReturnType<typeof conversations.conversation>>>();

const conversationQuery = (id: string, live = false) =>
  queryOptions({
    queryKey: [...conversationsKey, "one", id],
    queryFn: ({ signal }) => conversations.conversation(id, signal),
    // Read again on opening the page, coming back to the tab, and, while it isn't settled,
    // as Cayu's watches read (`watchInterval`), so an answer given in the inbox shows here.
    // Not on those while this page streams a reply in: the transcript is read when it ends,
    // as reading it sooner would show the message being answered twice. A page opened
    // again meanwhile reads it, since it shows no reply of its own beside it.
    staleTime: 0,
    refetchOnWindowFocus: !live,
    refetchInterval: live ? false : (query) => (UNSETTLED.has(query.state.data?.conversation.status ?? "") ? watchUnsettled(query) : false),
  });

function ConversationPage() {
  const { id = "" } = useParams();
  const queryClient = useQueryClient();
  const loaded = useQuery(conversationQuery(id, useStreaming(id)));
  // What went wrong with the reply that started this conversation, on the page that opened it.
  const started = useLocation().state as { error?: string } | null;
  const agent = useAgent();
  const { mayChange } = useWhoami();
  const notFound = isApiError(loaded.error, "not-found");
  return (
    <AppDetailPage data-origin="own">
      {notFound ? (
        <AppDetailHeader
          back={<AppBackLink render={<Link to="/conversations" />}>Back to conversations</AppBackLink>}
          title="Conversation not found"
          description="This conversation doesn't exist."
        />
      ) : (
        <AsyncContent {...queryContentProps(loaded, "Couldn't load this conversation.")} skeleton={<Skeleton className="h-96 w-full" />}>
          {loaded.data ? (
            <ConversationView
              id={loaded.data.conversation.id}
              title={titleOf(loaded.data.conversation)}
              status={loaded.data.conversation.status}
              startError={started?.error ?? null}
              messages={loaded.data.messages
                .map((item) => ({ role: item.role, text: textOf(item) }))
                .filter((item) => (item.role === "user" || item.role === "assistant") && item.text)}
              agent={agent ?? loaded.data.conversation.agent_name}
              mayContinue={mayChange}
              // Its transcript, and its place and last activity in the lists.
              onReplied={() => queryClient.invalidateQueries({ queryKey: conversationsKey })}
            />
          ) : null}
        </AsyncContent>
      )}
    </AppDetailPage>
  );
}

function ConversationView({
  id,
  title,
  status,
  startError,
  messages,
  agent,
  mayContinue,
  onReplied,
}: {
  id: string;
  title: string;
  status: string;
  startError: string | null;
  messages: { role: string; text: string }[];
  agent: string;
  mayContinue: boolean;
  onReplied: () => Promise<unknown>;
}) {
  const [text, setText] = React.useState("");
  // The message being answered and the reply so far, until the transcript has them.
  const [sent, setSent] = React.useState("");
  const [reply, setReply] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(startError);
  const scroller = React.useRef<HTMLDivElement>(null);
  // Follow new text only while the reader is at the bottom, not while they read back.
  const following = React.useRef(true);

  // Open at the latest message, and follow a reply as it streams.
  // It scrolls when any of these change.
  React.useEffect(() => {
    if (following.current) scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages, sent, reply, busy]);

  const queryClient = useQueryClient();
  const author = (role: string) => (role === "user" ? "Operator" : agent);
  const send = async () => {
    if (busy || !text.trim()) return;
    setBusy(true);
    setError(null);
    setSent(text);
    setText("");
    following.current = true;
    setStreaming(id, true);
    try {
      // A read already on its way could come back with the message in it.
      await queryClient.cancelQueries({ queryKey: conversationQuery(id).queryKey });
      await stream("resume", { session_id: id, prompt: text }, setReply);
    } catch (caught) {
      // Refused before it started: the message isn't in the conversation, so keep it.
      if (!(caught instanceof ReplyFailed)) setText(text);
      setError(messageOf(caught));
    } finally {
      setStreaming(id, false);
    }
    await onReplied();
    setSent("");
    setReply("");
    setBusy(false);
  };

  return (
    <div className="flex min-w-0 flex-col gap-page-section">
      <AppDetailHeader
        back={<AppBackLink render={<Link to="/conversations" />}>Back to conversations</AppBackLink>}
        title={title}
        badge={<StatusBadge tone={STATUS_TONE[status] ?? "neutral"}>{status}</StatusBadge>}
        description={`With ${agent}.`}
      />
      <div
        ref={scroller}
        onScroll={(event) => {
          const area = event.currentTarget;
          following.current = area.scrollHeight - area.scrollTop - area.clientHeight < 40;
        }}
        className="flex max-h-[60vh] min-h-48 flex-col gap-4 overflow-y-auto rounded-xl border bg-muted/30 p-4"
      >
        {messages.length === 0 && !sent ? <p className="text-body-sm text-text-secondary">No messages yet.</p> : null}
        {messages.map((item, index) => (
          // Transcript records have no ID; their order is fixed.
          // oxlint-disable-next-line react/no-array-index-key -- an append-only transcript
          <Bubble key={index} mine={item.role === "user"} author={author(item.role)} text={item.text} />
        ))}
        {sent ? <Bubble mine author={author("user")} text={sent} /> : null}
        {busy ? (
          reply ? (
            <Bubble mine={false} author={agent} text={reply} />
          ) : (
            <p className="animate-pulse text-body-sm text-text-secondary">{agent} is thinking…</p>
          )
        ) : null}
      </div>
      {mayContinue ? (
        <div className="flex items-end gap-2">
          <Textarea
            aria-label="Reply"
            placeholder="Reply… (Enter to send, Shift+Enter for a new line)"
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              // Not the Enter that confirms an input method's composition (Safari reports
              // that one with keyCode 229).
              const composing = event.nativeEvent.isComposing || event.keyCode === 229;
              if (event.key === "Enter" && !event.shiftKey && !composing) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <Button disabled={busy || !text.trim()} onClick={send}>
            {busy ? "Working…" : "Send"}
          </Button>
        </div>
      ) : (
        <p className="text-body-sm text-text-secondary">Your role reads conversations; an admin or a developer continues them.</p>
      )}
      {error ? <ErrorNote message={error} /> : null}
    </div>
  );
}

function Bubble({ mine, author, text }: { mine: boolean; author: string; text: string }) {
  return (
    <div className={mine ? "flex flex-col items-end gap-1" : "flex flex-col items-start gap-1"}>
      <span className="text-caption text-text-secondary">{author}</span>
      {mine ? (
        <div className="max-w-[80%] rounded-lg bg-primary px-3 py-2 whitespace-pre-wrap break-words text-primary-foreground">{text}</div>
      ) : (
        <div className="max-w-[80%] min-w-0 rounded-lg bg-background px-3 py-2 break-words ring-1 ring-border">
          <Markdown text={text} />
        </div>
      )}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <Alert variant="danger">
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
}
