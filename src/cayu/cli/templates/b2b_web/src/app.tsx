import { ArrowLeft } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  type Conversation,
  type Me,
  type Membership,
  type Message,
  ReplyFailed,
  type Role,
  type Sharing,
  stream,
  textOf,
  titleOf,
} from "./api";
import { Markdown } from "./components/markdown";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Input } from "./components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";
import { Textarea } from "./components/ui/textarea";

// Hash routes, so the server only serves one page: #/, #/new, #/c/<id>, #/members.
type Route =
  | { page: "conversations" }
  | { page: "new" }
  | { page: "conversation"; id: string }
  | { page: "members" };

function readRoute(): Route {
  const [, page, id] = window.location.hash.replace(/^#/, "").split("/");
  if (page === "new") return { page: "new" };
  if (page === "members") return { page: "members" };
  if (page === "c" && id) return { page: "conversation", id: decodeURIComponent(id) };
  return { page: "conversations" };
}

function go(hash: string) {
  window.location.hash = hash;
}

const ACTIVE_KEY = "active-organization";
const SELECT_CLASS =
  "h-8 rounded-lg border border-input bg-background px-2 text-sm outline-none focus:border-ring";
const ROLES: Role[] = ["member", "admin"];
const SHARING_LABEL: Record<Sharing, string> = {
  personal: "Only me",
  department: "My department",
  organization: "My organization",
};

export function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [signedOut, setSignedOut] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [route, setRoute] = useState<Route>(readRoute);
  const [active, setActive] = useState<string | null>(() => localStorage.getItem(ACTIVE_KEY));
  // The demonstration people, offered only under `cayu serve --dev`.
  const [people, setPeople] = useState<string[] | null>(null);
  // An error that leaves the page usable.
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setMe(await api.me());
      setSignedOut(false);
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) setSignedOut(true);
      else setError(message(caught));
    }
  }, []);

  useEffect(() => {
    void load();
    api
      .demoPeople()
      .then(setPeople)
      .catch(() => setPeople([]));
    const onHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [load]);

  if (error) return <Centered title="Something went wrong" description={error} />;
  if (signedOut && people !== null) return <SignIn people={people} onSignedIn={load} />;
  if (!me) return <Centered title="Loading…" />;
  if (me.memberships.length === 0) {
    return (
      <Centered
        title="No organization yet"
        description={`You're signed in as ${me.subject}, but no organization has added you.`}
      />
    );
  }

  // The active organization is only a filter and a default; the server decides access.
  const membership: Membership =
    me.memberships.find((item) => item.organization === active) ??
    (me.memberships[0] as Membership);
  const choose = (organization: string) => {
    localStorage.setItem(ACTIVE_KEY, organization);
    setActive(organization);
    go("#/");
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3">
        <a href="#/" className="font-semibold">
          {me.agent_name}
        </a>
        <select
          aria-label="Active organization"
          className={SELECT_CLASS}
          value={membership.organization}
          onChange={(event) => choose(event.target.value)}
        >
          {me.memberships.map((item) => (
            <option key={item.organization} value={item.organization}>
              {item.organization}
              {item.department ? ` · ${item.department}` : ""}
            </option>
          ))}
        </select>
        <nav className="ml-auto flex min-w-0 flex-wrap items-center gap-2 text-sm">
          <Button variant="ghost" onClick={() => go("#/")}>
            Conversations
          </Button>
          {membership.role === "admin" && (
            <Button variant="ghost" onClick={() => go("#/members")}>
              Members
            </Button>
          )}
          <span className="max-w-40 truncate text-muted-foreground">{me.subject}</span>
          {people && people.length > 0 && (
            <Button
              variant="outline"
              onClick={() =>
                api
                  .demoSignOut()
                  .then(() => {
                    setNotice(null);
                    setMe(null);
                    setSignedOut(true);
                  })
                  .catch((caught) => setNotice(message(caught)))
              }
            >
              Switch person
            </Button>
          )}
        </nav>
      </header>
      <main className="mx-auto max-w-4xl p-4">
        {notice && <p className="mb-3 text-destructive">{notice}</p>}
        {/* Keyed by what each page shows, so switching starts it afresh and a late
            answer for the previous one is dropped. */}
        {route.page === "conversations" && (
          <Conversations key={membership.organization} membership={membership} />
        )}
        {route.page === "new" && <NewConversation me={me} membership={membership} />}
        {route.page === "conversation" && <ConversationView key={route.id} me={me} id={route.id} />}
        {route.page === "members" && (
          <Members key={membership.organization} membership={membership} onChanged={load} />
        )}
      </main>
    </div>
  );
}

function SignIn({ people, onSignedIn }: { people: string[]; onSignedIn: () => void }) {
  const [error, setError] = useState<string | null>(null);

  if (people.length === 0) {
    return (
      <Centered
        title="Sign in"
        description="Sign in through your organization's sign-in, then open this page again."
      />
    );
  }
  return (
    <Centered
      title="Sign in"
      description="Development only: choose a demonstration person."
      error={error}
    >
      <div className="flex flex-wrap gap-2">
        {people.map((subject) => (
          <Button
            key={subject}
            variant="outline"
            onClick={() =>
              api
                .demoSignIn(subject)
                .then(onSignedIn)
                .catch((caught) => setError(message(caught)))
            }
          >
            {subject}
          </Button>
        ))}
      </div>
    </Centered>
  );
}

function Conversations({ membership }: { membership: Membership }) {
  const [items, setItems] = useState<Conversation[] | null>(null);
  // Where the next page starts, while there is one.
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const more = useCallback(
    async (from: string | null) => {
      setLoading(true);
      try {
        const page = await api.conversations(membership.organization, from);
        setError(null);
        setItems((shown) => [...(from === null ? [] : (shown ?? [])), ...page.sessions]);
        setCursor(page.next_cursor);
      } catch (caught) {
        setError(message(caught));
      } finally {
        setLoading(false);
      }
    },
    [membership.organization],
  );

  useEffect(() => {
    void more(null);
  }, [more]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Conversations in {membership.organization}</CardTitle>
        <CardDescription>Your own, and those shared with you.</CardDescription>
        <CardAction>
          <Button onClick={() => go("#/new")}>New conversation</Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {error ? (
          <p className="text-destructive">{error}</p>
        ) : items === null ? (
          <p className="text-muted-foreground">Loading…</p>
        ) : items.length === 0 ? (
          <p className="text-muted-foreground">No conversations you can see here yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Conversation</TableHead>
                <TableHead>Shared with</TableHead>
                <TableHead>Owner</TableHead>
                <TableHead>Last activity</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((item) => (
                <TableRow
                  key={item.id}
                  className="cursor-pointer"
                  onClick={() => go(`#/c/${encodeURIComponent(item.id)}`)}
                >
                  <TableCell>{titleOf(item)}</TableCell>
                  <TableCell>
                    <SharingBadge labels={item.labels} />
                  </TableCell>
                  <TableCell>{item.labels.owner}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {when(item.last_activity_at)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        {cursor && (
          <div className="mt-3 flex justify-center">
            <Button variant="outline" disabled={loading} onClick={() => more(cursor)}>
              {loading ? "Loading…" : "Show more"}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function NewConversation({ me, membership }: { me: Me; membership: Membership }) {
  const [sharing, setSharing] = useState<Sharing>("personal");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [reply, setReply] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const options: Sharing[] = membership.department
    ? ["personal", "department", "organization"]
    : ["personal", "organization"];
  // Leaving the page keeps reading the reply, since stopping would stop the agent, but it no
  // longer updates this page or opens the conversation.
  const shown = useRef(true);
  useEffect(() => {
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
        {
          agent_name: me.agent_name,
          title: title.trim() || undefined,
          labels,
          messages: [{ role: "user", content: [{ type: "text", text }] }],
        },
        (text) => shown.current && setReply(text),
      );
      if (shown.current) go(`#/c/${encodeURIComponent(id)}`);
    } catch (caught) {
      if (!shown.current) return;
      // Once it has started, the conversation exists: open it rather than start another.
      if (caught instanceof ReplyFailed && caught.sessionId) {
        go(`#/c/${encodeURIComponent(caught.sessionId)}`);
        return;
      }
      setError(message(caught));
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>New conversation</CardTitle>
        <CardDescription>In {membership.organization}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <fieldset className="flex flex-wrap gap-2">
          <legend className="mb-1 text-sm text-muted-foreground">Who can see it</legend>
          {options.map((option) => (
            <Button
              key={option}
              variant={option === sharing ? "default" : "outline"}
              aria-pressed={option === sharing}
              onClick={() => setSharing(option)}
            >
              {SHARING_LABEL[option]}
            </Button>
          ))}
        </fieldset>
        <Input
          aria-label="Title"
          placeholder="Title (optional)"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
        />
        <Textarea
          aria-label="Message"
          placeholder="What should the agent do?"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <div>
          <Button disabled={busy || !text.trim()} onClick={start}>
            {busy ? "Working…" : "Start"}
          </Button>
        </div>
        {reply && (
          <div className="rounded-md bg-muted p-3">
            <Markdown text={reply} />
          </div>
        )}
        {error && <p className="text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}

function ConversationView({ me, id }: { me: Me; id: string }) {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [text, setText] = useState("");
  // The message being answered and the reply so far, until the transcript has them.
  const [sent, setSent] = useState("");
  const [reply, setReply] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  // Follow new text only while the reader is at the bottom, not while they read back.
  const following = useRef(true);

  const load = useCallback(async () => {
    try {
      setConversation(await api.conversation(id));
      setMessages(await api.transcript(id));
    } catch (caught) {
      setError(
        caught instanceof ApiError && caught.status === 404
          ? "This conversation doesn't exist, or you can't see it."
          : message(caught),
      );
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  // Open at the latest message, and follow a reply as it streams.
  // biome-ignore lint/correctness/useExhaustiveDependencies: scrolls when any of these change
  useEffect(() => {
    if (following.current) scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages, sent, reply, busy]);

  if (error && !conversation) return <p className="text-destructive">{error}</p>;
  if (!conversation) return <p className="text-muted-foreground">Loading…</p>;

  // Only its owner continues a conversation; others may only read a shared one.
  const ownerName = conversation.labels.owner ?? "its owner";
  const owner = ownerName === me.subject;
  const author = (role: string) => (role === "user" ? (owner ? "You" : ownerName) : me.agent_name);
  const send = async () => {
    if (busy || !text.trim()) return;
    setBusy(true);
    setError(null);
    setSent(text);
    setText("");
    following.current = true;
    try {
      await stream(
        "resources/resume",
        { session_id: id, messages: [{ role: "user", content: [{ type: "text", text }] }] },
        setReply,
      );
    } catch (caught) {
      // Refused before it started: the message isn't in the conversation, so keep it.
      if (!(caught instanceof ReplyFailed)) setText(text);
      setError(message(caught));
    }
    await load();
    setSent("");
    setReply("");
    setBusy(false);
  };
  const shown = messages
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item) => ({ role: item.role, text: textOf(item) }))
    .filter((item) => item.text);

  return (
    <Card>
      <CardHeader>
        <CardTitle>{titleOf(conversation)}</CardTitle>
        <CardDescription className="flex flex-wrap items-center gap-2">
          <SharingBadge labels={conversation.labels} />
          <span>Started by {owner ? "you" : ownerName}</span>
        </CardDescription>
        <CardAction>
          <Button variant="outline" onClick={() => go("#/")}>
            <ArrowLeft />
            All conversations
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="flex min-w-0 flex-col gap-3">
        <div
          ref={scroller}
          onScroll={(event) => {
            const area = event.currentTarget;
            following.current = area.scrollHeight - area.scrollTop - area.clientHeight < 40;
          }}
          className="flex max-h-[60vh] min-h-48 flex-col gap-4 overflow-y-auto rounded-lg border border-border bg-muted/30 p-4"
        >
          {shown.map((item, index) => (
            <Bubble
              // Transcript records have no ID; their order is fixed.
              // biome-ignore lint/suspicious/noArrayIndexKey: an append-only transcript
              key={index}
              mine={item.role === "user"}
              author={author(item.role)}
              text={item.text}
            />
          ))}
          {sent && <Bubble mine author={author("user")} text={sent} />}
          {busy &&
            (reply ? (
              <Bubble mine={false} author={me.agent_name} text={reply} />
            ) : (
              <p className="animate-pulse text-sm text-muted-foreground">
                {me.agent_name} is thinking…
              </p>
            ))}
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
          <p className="text-sm text-muted-foreground">
            Shared with you to read; only {ownerName} can continue it.
          </p>
        )}
        {error && <p className="text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}

function Bubble({ mine, author, text }: { mine: boolean; author: string; text: string }) {
  return (
    <div className={mine ? "flex flex-col items-end gap-1" : "flex flex-col items-start gap-1"}>
      <span className="text-xs text-muted-foreground">{author}</span>
      {mine ? (
        <div className="max-w-[80%] whitespace-pre-wrap break-words rounded-lg bg-primary px-3 py-2 text-primary-foreground">
          {text}
        </div>
      ) : (
        <div className="min-w-0 max-w-[80%] break-words rounded-lg bg-background px-3 py-2 ring-1 ring-border">
          <Markdown text={text} />
        </div>
      )}
    </div>
  );
}

function Members({ membership, onChanged }: { membership: Membership; onChanged: () => void }) {
  const organization = membership.organization;
  // The member whose removal is waiting for a second click.
  const [confirming, setConfirming] = useState<string | null>(null);
  const [members, setMembers] = useState<Membership[] | null>(null);
  const [subject, setSubject] = useState("");
  const [department, setDepartment] = useState("");
  const [role, setRole] = useState<Role>("member");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api
      .members(organization)
      .then(setMembers)
      .catch((caught) => setError(message(caught)));
  }, [organization]);

  useEffect(load, [load]);

  const change = async (action: Promise<unknown>, onDone?: () => void) => {
    try {
      await action;
      setError(null);
      onDone?.();
      load();
      // An admin may have changed their own role.
      onChanged();
    } catch (caught) {
      setError(message(caught));
    }
  };

  if (membership.role !== "admin") {
    return <p className="text-muted-foreground">Only the organization's admins manage members.</p>;
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Members of {organization}</CardTitle>
        <CardDescription>Removing someone takes effect at their next step.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Person</TableHead>
              <TableHead>Department</TableHead>
              <TableHead>Role</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {(members ?? []).map((item) => (
              <TableRow key={item.subject}>
                <TableCell>{item.subject}</TableCell>
                <TableCell>{item.department ?? "—"}</TableCell>
                <TableCell>
                  <Badge variant={item.role === "admin" ? "default" : "secondary"}>
                    {item.role}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">
                  {confirming === item.subject ? (
                    <div className="flex justify-end gap-2">
                      <Button variant="ghost" onClick={() => setConfirming(null)}>
                        Cancel
                      </Button>
                      <Button
                        variant="destructive"
                        onClick={() =>
                          change(api.removeMember(organization, item.subject), () =>
                            setConfirming(null),
                          )
                        }
                      >
                        Confirm remove
                      </Button>
                    </div>
                  ) : (
                    <Button variant="destructive" onClick={() => setConfirming(item.subject)}>
                      Remove
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="flex flex-wrap items-end gap-2">
          <Input
            aria-label="Person"
            placeholder="Person's ID"
            className="w-48"
            value={subject}
            onChange={(event) => setSubject(event.target.value)}
          />
          <Input
            aria-label="Department"
            placeholder={`Department, e.g. ${organization}/finance`}
            className="w-64"
            value={department}
            onChange={(event) => setDepartment(event.target.value)}
          />
          {ROLES.map((option) => (
            <Button
              key={option}
              variant={option === role ? "default" : "outline"}
              aria-pressed={option === role}
              onClick={() => setRole(option)}
            >
              {option}
            </Button>
          ))}
          <Button
            disabled={!subject.trim()}
            onClick={() =>
              change(
                api.putMember(organization, subject.trim(), department.trim() || null, role),
                () => {
                  setSubject("");
                  setDepartment("");
                  setRole("member");
                },
              )
            }
          >
            Add or update
          </Button>
        </div>
        {error && <p className="text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}

function SharingBadge({ labels }: { labels: Record<string, string> }) {
  const sharing = labels.sharing as Sharing | undefined;
  const text =
    sharing === "department"
      ? labels.department
      : sharing === "organization"
        ? labels.organization
        : "Only the owner";
  return <Badge variant="outline">{text}</Badge>;
}

function Centered({
  title,
  description,
  error,
  children,
}: {
  title: string;
  description?: string;
  error?: string | null;
  children?: ReactNode;
}) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4 text-foreground">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        {(children || error) && (
          <CardContent className="flex flex-col gap-3">
            {children}
            {error && <p className="text-destructive">{error}</p>}
          </CardContent>
        )}
      </Card>
    </div>
  );
}

function message(caught: unknown): string {
  return caught instanceof Error ? caught.message : String(caught);
}

function when(timestamp: string | null): string {
  return timestamp ? new Date(timestamp).toLocaleString() : "";
}
