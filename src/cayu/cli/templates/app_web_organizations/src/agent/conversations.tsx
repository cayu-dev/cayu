import * as React from "react";
import { Link, useNavigate, useParams } from "react-router";
import { queryOptions, useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import { AppBackLink, AppDetailHeader, AppDetailPage, AppPageHeader, AppPageTitle } from "@kit/layouts/app";
import { AsyncContent, DataTable } from "@kit/patterns";
import { Alert, AlertDescription, Button, Card, CardContent, Field, FieldLabel, Input, Skeleton, StatusBadge, Textarea, ToggleGroup, ToggleGroupItem } from "@kit/ui";
import { formatDateTime, formatRelative } from "@kit/lib";
import { queryContentProps, queryTableProps } from "@kit/data";
import { BubbleChatIcon } from "@kit/icons";
import type { AppArea } from "../app/area";
import { useAreaApi } from "../app/services";
import { b2bApi, type B2bApi, type Conversation, type Membership, messageOf, ReplyFailed, type Sharing, statusOf, stream, textOf, titleOf } from "./api";
import { Markdown } from "./markdown";
import { MembershipFailed, membershipBlocker, OrganizationSelect, useMembership } from "./organization";

/**
 * THE AGENT'S OWN. Conversations with the agent: the ones the person can see in their active
 * organization, a new one shared as they choose, and one conversation, which only its owner
 * continues.
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
export const conversationsKey = ["b2b", "conversations"] as const;

export const SHARING_LABEL: Record<Sharing, string> = {
  personal: "Only me",
  department: "My department",
  organization: "My organization",
};

/** Who can see a conversation, from its sharing labels. */
export function SharingBadge({ labels }: { labels: Record<string, string> }) {
  const text = labels.sharing === "department" ? labels.department : labels.sharing === "organization" ? labels.organization : "Only the owner";
  return <StatusBadge tone="neutral">{text ?? "Unknown"}</StatusBadge>;
}

export const conversationsQuery = (api: B2bApi, organization: string | null, pageSize = 50) =>
  queryOptions({
    queryKey: [...conversationsKey, organization, pageSize],
    queryFn: ({ signal }) => api.conversations(organization, null, pageSize, signal),
  });

function ConversationsPage() {
  const { me, membership } = useMembership();
  const api = useAreaApi(b2bApi);
  const organization = membership?.organization ?? null;
  // Pages of 50, the next one asked for only when the person wants more.
  const pages = useInfiniteQuery({
    queryKey: [...conversationsKey, organization, "all"],
    queryFn: ({ pageParam, signal }) => api.conversations(organization, pageParam, 50, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor,
    enabled: organization !== null,
  });
  const rows = pages.data?.pages.flatMap((page) => page.sessions) ?? [];
  const blocked = membershipBlocker(me, membership);
  if (blocked) return blocked;
  return (
    <div className="flex flex-col gap-page-section" data-origin="own">
      <AppPageHeader
        description={organization ? `In ${organization}: your own, and those shared with your department or organization.` : null}
        actions={
          <>
            <OrganizationSelect />
            <Button render={<Link to="/conversations/new" />}>New conversation</Button>
          </>
        }
      >
        <AppPageTitle>Conversations</AppPageTitle>
      </AppPageHeader>
      <DataTable
        {...queryTableProps(pages, "Couldn't load the conversations.")}
        label="Conversations"
        showHeading={false}
        rows={rows}
        rowId={(row) => row.id}
        searchText={(row) => `${titleOf(row)} ${row.labels.owner ?? ""}`}
        itemLabel={{ singular: "conversation", plural: "conversations" }}
        emptyTitle="No conversations yet"
        emptyDescription="None you can see here yet. Start one, or ask someone to share theirs."
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
          { key: "sharing", label: "Shared with", cell: (row) => <SharingBadge labels={row.labels} /> },
          { key: "owner", label: "Owner", sortValue: (row) => row.labels.owner ?? "", cell: (row) => row.labels.owner ?? "—" },
          {
            key: "activity",
            label: "Last activity",
            sortValue: (row) => row.last_activity_at ?? "",
            cell: (row) => (
              <time dateTime={row.last_activity_at ?? undefined} title={formatDateTime(row.last_activity_at)}>
                {formatRelative(row.last_activity_at)}
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
  const { me, membership } = useMembership();
  const blocked = membershipBlocker(me, membership);
  if (blocked) return blocked;
  return (
    <AppDetailPage data-origin="own">
      <AppDetailHeader
        back={<AppBackLink render={<Link to="/conversations" />}>Back to conversations</AppBackLink>}
        title="New conversation"
        description={membership ? `In ${membership.organization}. You choose who else can read it; only you continue it.` : null}
        actions={<OrganizationSelect />}
      />
      {membership && me.data ? <NewConversationForm key={membership.organization} agent={me.data.agent_name} membership={membership} /> : <Skeleton className="h-64 w-full" />}
    </AppDetailPage>
  );
}

function NewConversationForm({ agent, membership }: { agent: string; membership: Membership }) {
  const api = useAreaApi(b2bApi);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [sharing, setSharing] = React.useState<Sharing>("personal");
  const [title, setTitle] = React.useState("");
  const [text, setText] = React.useState("");
  const [reply, setReply] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const options: Sharing[] = membership.department ? ["personal", "department", "organization"] : ["personal", "organization"];
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
      // The server builds the labels from your membership; it never takes them from here.
      const labels = await api.labels(membership.organization, sharing);
      const id = await stream(
        "resources/runs",
        { agent_name: agent, title: title.trim() || undefined, labels, messages: [{ role: "user", content: [{ type: "text", text }] }] },
        (next) => shown.current && setReply(next),
      );
      // The lists don't have it yet, whether or not this page is still shown.
      void queryClient.invalidateQueries({ queryKey: conversationsKey });
      if (shown.current) navigate(conversationPath(id));
    } catch (caught) {
      // Once it has started, the conversation exists: open it rather than start another.
      if (caught instanceof ReplyFailed && caught.sessionId) {
        void queryClient.invalidateQueries({ queryKey: conversationsKey });
        if (shown.current) navigate(conversationPath(caught.sessionId));
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
          <FieldLabel>Who can read it</FieldLabel>
          <ToggleGroup
            variant="outline"
            spacing={1}
            value={[sharing]}
            onValueChange={(next: string[]) => {
              const value = next.at(-1);
              if (value) setSharing(value as Sharing);
            }}
            className="flex-wrap"
          >
            {options.map((option) => (
              <ToggleGroupItem key={option} value={option}>
                {SHARING_LABEL[option]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </Field>
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

const conversationQuery = (api: B2bApi, id: string) =>
  queryOptions({
    queryKey: ["b2b", "conversation", id],
    queryFn: async ({ signal }) => {
      const found = await api.conversation(id, signal);
      return { conversation: found, messages: await api.transcript(id, signal) };
    },
    // A reply that streams in is followed live; the transcript is read again when it ends.
    staleTime: 0,
  });

function ConversationPage() {
  const { id = "" } = useParams();
  const api = useAreaApi(b2bApi);
  const queryClient = useQueryClient();
  const loaded = useQuery(conversationQuery(api, id));
  const { me } = useMembership();
  const notFound = statusOf(loaded.error) === 404;
  return (
    <AppDetailPage data-origin="own">
      {notFound ? (
        <AppDetailHeader
          back={<AppBackLink render={<Link to="/conversations" />}>Back to conversations</AppBackLink>}
          title="Conversation not found"
          description="This conversation doesn't exist, or you can't see it."
        />
      ) : me.isError && me.data === undefined ? (
        <MembershipFailed message={messageOf(me.error)} retry={() => void me.refetch()} />
      ) : (
        <AsyncContent {...queryContentProps(loaded, "Couldn't load this conversation.")} skeleton={<Skeleton className="h-96 w-full" />}>
          {loaded.data && me.data ? (
            <ConversationView
              conversation={loaded.data.conversation}
              messages={loaded.data.messages.map((item) => ({ role: item.role, text: textOf(item) })).filter((item) => (item.role === "user" || item.role === "assistant") && item.text)}
              subject={me.data.subject}
              agent={me.data.agent_name}
              onReplied={() =>
                // Its transcript, and its place and last activity in the lists.
                Promise.all([loaded.refetch(), queryClient.invalidateQueries({ queryKey: conversationsKey })])
              }
            />
          ) : null}
        </AsyncContent>
      )}
    </AppDetailPage>
  );
}

function ConversationView({
  conversation,
  messages,
  subject,
  agent,
  onReplied,
}: {
  conversation: Conversation;
  messages: { role: string; text: string }[];
  subject: string;
  agent: string;
  onReplied: () => Promise<unknown>;
}) {
  const [text, setText] = React.useState("");
  // The message being answered and the reply so far, until the transcript has them.
  const [sent, setSent] = React.useState("");
  const [reply, setReply] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const scroller = React.useRef<HTMLDivElement>(null);
  // Follow new text only while the reader is at the bottom, not while they read back.
  const following = React.useRef(true);

  // Open at the latest message, and follow a reply as it streams.
  // biome-ignore lint/correctness/useExhaustiveDependencies: scrolls when any of these change
  React.useEffect(() => {
    if (following.current) scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages, sent, reply, busy]);

  // Only its owner continues a conversation; others may only read a shared one.
  const ownerName = conversation.labels.owner ?? "its owner";
  const owner = ownerName === subject;
  const author = (role: string) => (role === "user" ? (owner ? "You" : ownerName) : agent);
  const send = async () => {
    if (busy || !text.trim()) return;
    setBusy(true);
    setError(null);
    setSent(text);
    setText("");
    following.current = true;
    try {
      await stream("resources/resume", { session_id: conversation.id, messages: [{ role: "user", content: [{ type: "text", text }] }] }, setReply);
    } catch (caught) {
      // Refused before it started: the message isn't in the conversation, so keep it.
      if (!(caught instanceof ReplyFailed)) setText(text);
      setError(messageOf(caught));
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
        title={titleOf(conversation)}
        badge={<SharingBadge labels={conversation.labels} />}
        description={`Started by ${owner ? "you" : ownerName}${conversation.labels.organization ? ` in ${conversation.labels.organization}` : ""}.`}
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
          // biome-ignore lint/suspicious/noArrayIndexKey: an append-only transcript
          <Bubble key={index} mine={item.role === "user"} author={author(item.role)} text={item.text} />
        ))}
        {sent ? <Bubble mine author={author("user")} text={sent} /> : null}
        {busy ? reply ? <Bubble mine={false} author={agent} text={reply} /> : <p className="animate-pulse text-body-sm text-text-secondary">{agent} is thinking…</p> : null}
      </div>
      {owner ? (
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
        <p className="text-body-sm text-text-secondary">Shared with you to read; only {ownerName} can continue it.</p>
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
