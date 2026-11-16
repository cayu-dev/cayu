import * as React from "react";
import { Link, useSearchParams } from "react-router";
import { keepPreviousData, queryOptions, useQuery, useQueryClient } from "@tanstack/react-query";
import { AppPageHeader, AppPageSection, AppPageTitle } from "@kit/layouts/app";
import { AsyncContent } from "@kit/patterns";
import {
  Alert,
  AlertDescription,
  Badge,
  Button,
  Card,
  CardContent,
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  Input,
  Skeleton,
  Textarea,
  toast,
} from "@kit/ui";
import { formatDateTime, formatRelative, plural } from "@kit/lib";
import { queryContentProps, watchInterval } from "@kit/data";
import {
  BookOpen01Icon,
  BubbleChatIcon,
  Delete02Icon,
  File01Icon,
  HugeiconsIcon,
  type IconSvgElement,
  Idea01Icon,
  PencilEdit02Icon,
  Search01Icon,
  SparklesIcon,
  UserIcon,
} from "@kit/icons";
import { type AppArea, areaPath } from "../../app/area";
import { useAreaApi } from "../../app/services";
import { type Knowledge, type KnowledgeApi, type KnowledgeSource, knowledgeApi, type Statement, type Suggestion, type Suggestions } from "../../api/knowledge";

/** How many statements a read asks for, and how many more each "Show more" adds. */
const PAGE = 50;
const MOST = 500;
/** How long typing pauses before the search runs. */
const SEARCH_PAUSE_MS = 300;

// Followed as Cayu's watches follow its API, with no timer of its own: every 15 seconds while
// it changes, less often while it doesn't. The routes tag their answers, so an unchanged read
// costs a 304. A new search keeps the last answer on screen until the new one arrives.
export const knowledgeQuery = (api: KnowledgeApi, q: string, limit: number) =>
  queryOptions({
    queryKey: ["knowledge", "statements", q, limit],
    queryFn: ({ signal }) => api.list(q, limit, signal),
    refetchInterval: watchInterval<Knowledge>(),
    placeholderData: keepPreviousData,
  });

export const suggestionsQuery = (api: KnowledgeApi) =>
  queryOptions({
    queryKey: ["knowledge", "suggestions"],
    queryFn: ({ signal }) => api.suggestions(signal),
    refetchInterval: watchInterval<Suggestions>(),
  });

/**
 * GENERATED. The Knowledge area: what the Agent knows, written as plain statements, where each
 * came from and when it was learned, and what it asked to remember. People whose role may
 * change things correct a statement or have it forgotten. The Developer dashboard (/cayu/) keeps
 * the full view, with revisions, sources and recall evidence.
 */
export const knowledgeArea: AppArea = {
  id: "knowledge",
  label: "Knowledge",
  icon: BookOpen01Icon,
  path: "/knowledge",
  routes: ({ areas }) => [{ path: "knowledge", element: <KnowledgePage conversations={areaPath(areas, "conversations")} /> }],
};

function KnowledgePage({ conversations }: { conversations: string | null }) {
  const api = useAreaApi(knowledgeApi);
  const [params, setParams] = useSearchParams();
  const q = params.get("q")?.trim() ?? "";
  const [draft, setDraft] = React.useState(q);
  const [limit, setLimit] = React.useState(PAGE);
  const known = useQuery(knowledgeQuery(api, q, limit));
  const suggested = useQuery(suggestionsQuery(api));

  // The search reaches the URL, and the server, once typing pauses.
  React.useEffect(() => {
    const next = draft.trim();
    if (next === q) return;
    const timer = window.setTimeout(() => {
      setLimit(PAGE);
      setParams(
        (current) => {
          const updated = new URLSearchParams(current);
          if (next) updated.set("q", next);
          else updated.delete("q");
          return updated;
        },
        { replace: true },
      );
    }, SEARCH_PAUSE_MS);
    return () => window.clearTimeout(timer);
  }, [draft, q, setParams]);

  return (
    <div className="flex flex-col gap-page-section" data-origin="generated">
      <AppPageHeader description="What the Agent knows and uses in its work: where it learned each thing, and what it would like to remember.">
        <AppPageTitle>Knowledge</AppPageTitle>
      </AppPageHeader>

      <div className="relative max-w-xl">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={2}
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          type="search"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Search what the Agent knows"
          aria-label="Search what the Agent knows"
          className="h-10 pl-9"
        />
      </div>

      {suggested.data && suggested.data.suggestions.length > 0 && !q ? <SuggestionList data={suggested.data} conversations={conversations} /> : null}

      <AppPageSection
        title={q ? `Matching “${q}”` : "What the Agent knows"}
        description={known.data && known.data.statements.length > 0 ? statementCount(known.data, q) : undefined}
      >
        <AsyncContent {...queryContentProps(known, "Couldn't load what the Agent knows.")} skeleton={<StatementsSkeleton />}>
          {known.data ? (
            known.data.statements.length === 0 ? (
              q ? (
                <NoMatch q={q} onClear={() => setDraft("")} />
              ) : (
                <NothingYet />
              )
            ) : (
              <div className="flex flex-col gap-4">
                <Card className="py-0">
                  <ul className="divide-y">
                    {known.data.statements.map((item) => (
                      <StatementRow key={item.id} item={item} conversations={conversations} />
                    ))}
                  </ul>
                </Card>
                {known.data.more && limit < MOST ? (
                  <Button
                    variant="outline"
                    className="self-center"
                    disabled={known.isFetching}
                    onClick={() => setLimit((current) => Math.min(MOST, current + PAGE))}
                  >
                    Show more
                  </Button>
                ) : null}
              </div>
            )
          ) : null}
        </AsyncContent>
      </AppPageSection>
    </div>
  );
}

function statementCount(data: Knowledge, q: string): string {
  const count = plural(data.statements.length, "statement");
  if (q) return data.more ? `The best ${count} that match.` : `${count} ${data.statements.length === 1 ? "matches" : "match"}.`;
  // Latest first: a statement someone corrected comes up with the newly learned ones.
  return data.more ? `The latest ${count}.` : `${count}, latest first.`;
}

function SuggestionList({ data, conversations }: { data: Suggestions; conversations: string | null }) {
  return (
    <AppPageSection
      title={
        <span className="flex items-center gap-2">
          Waiting to be remembered
          <Badge variant="secondary">{data.more ? `${data.suggestions.length}+` : data.suggestions.length}</Badge>
        </span>
      }
      description={`Things the Agent learned and would like to keep. It doesn't use them until someone keeps them. ${data.reviewed_by}`}
    >
      <ul className="grid gap-3 md:grid-cols-2">
        {data.suggestions.map((item) => (
          <SuggestionCard key={item.id} item={item} conversations={conversations} />
        ))}
      </ul>
    </AppPageSection>
  );
}

function SuggestionCard({ item, conversations }: { item: Suggestion; conversations: string | null }) {
  return (
    <li className="flex flex-col gap-3 rounded-xl border border-dashed bg-muted/30 p-4">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-background text-muted-foreground ring-1 ring-border">
          <HugeiconsIcon icon={Idea01Icon} strokeWidth={2} className="size-3.5" aria-hidden="true" />
        </span>
        <p className="min-w-0 text-sm leading-relaxed break-words whitespace-pre-wrap">{item.text}</p>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 pl-10">
        <Meta>
          <SourceNote source={item.source} conversations={conversations} />
          <Dot />
          <Moment verb="Suggested" at={item.suggested_at} />
          {item.group ? <GroupBadge group={item.group} /> : null}
        </Meta>
        {item.review ? (
          <Button variant="outline" size="sm" render={<a href={item.review} />} nativeButton={false}>
            Review in the Inbox
          </Button>
        ) : null}
      </div>
    </li>
  );
}

function StatementRow({ item, conversations }: { item: Statement; conversations: string | null }) {
  const api = useAreaApi(knowledgeApi);
  const queryClient = useQueryClient();
  const [mode, setMode] = React.useState<"view" | "correct" | "forget">("view");
  // The statement as it was when the person opened Correct or Forget. The page reads it again
  // in the background; a change names this version, so one made by someone else meanwhile
  // is refused (409) rather than overwritten.
  const [opened, setOpened] = React.useState<Statement>(item);
  const [text, setText] = React.useState(item.text);
  const [reason, setReason] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const change = async (action: () => Promise<unknown>, done: string) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      toast.add({ title: done, type: "success" });
      setMode("view");
      await queryClient.invalidateQueries({ queryKey: ["knowledge"] });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };
  const open = (next: "correct" | "forget") => {
    setOpened(item);
    setText(item.text);
    setReason("");
    setError(null);
    setMode(next);
  };

  return (
    <li className="flex flex-col gap-3 px-5 py-4">
      {mode === "correct" ? (
        <div className="flex flex-col gap-2">
          {/* oxlint-disable-next-line jsx-a11y/no-autofocus -- Focus follows the person choosing Correct; this editor never mounts on initial page load. */}
          <Textarea value={text} onChange={(event) => setText(event.target.value)} rows={3} aria-label="The corrected statement" autoFocus />
          <p className="text-xs text-muted-foreground">The Agent uses your wording from now on. Where it learned this stays as it was.</p>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              disabled={busy || !text.trim() || text.trim() === opened.text}
              onClick={() => change(() => api.correct(opened, text.trim()), "Corrected")}
            >
              Save correction
            </Button>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => setMode("view")}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex items-start justify-between gap-4">
          <p className="min-w-0 text-[0.95rem] leading-relaxed break-words whitespace-pre-wrap">{item.text}</p>
          {item.may_change && mode === "view" ? (
            <div className="-mt-1 -mr-2 flex shrink-0 gap-1 text-muted-foreground">
              {item.may_correct ? (
                <Button variant="ghost" size="sm" aria-label={`Correct: ${item.text}`} onClick={() => open("correct")}>
                  <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={2} data-icon="inline-start" />
                  Correct
                </Button>
              ) : null}
              <Button variant="ghost" size="sm" aria-label={`Forget: ${item.text}`} onClick={() => open("forget")}>
                <HugeiconsIcon icon={Delete02Icon} strokeWidth={2} data-icon="inline-start" />
                Forget
              </Button>
            </div>
          ) : null}
        </div>
      )}
      <Meta>
        <SourceNote source={item.source} conversations={conversations} />
        <Dot />
        <Moment verb="Learned" at={item.learned_at} />
        {item.corrected ? (
          <>
            <Dot />
            <Moment verb={`Corrected by ${item.corrected.by}`} at={item.corrected.at} />
          </>
        ) : null}
        {item.group ? <GroupBadge group={item.group} /> : null}
      </Meta>
      {mode === "forget" ? (
        <div className="flex flex-col gap-3 rounded-lg bg-muted/50 p-3">
          <p className="text-sm">Forget this? The Agent stops using it right away. Its builders can still find it in the Developer dashboard's archive.</p>
          <Input
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="Why, if you'd like to say"
            aria-label="Why it should be forgotten"
            maxLength={500}
          />
          <div className="flex flex-wrap gap-2">
            {/* The default variant: kits name their danger variants differently. */}
            <Button size="sm" disabled={busy} onClick={() => change(() => api.forget(opened, reason.trim()), "Forgotten")}>
              Forget it
            </Button>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => setMode("view")}>
              Keep it
            </Button>
          </div>
        </div>
      ) : null}
      {mode !== "view" && item.version !== opened.version ? (
        <Alert variant="warning">
          <AlertDescription>Someone changed this statement since you opened it, so this change would be refused. Cancel to see the latest.</AlertDescription>
        </Alert>
      ) : null}
      {error ? (
        <Alert variant="danger">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}
    </li>
  );
}

const SOURCE_ICONS: Record<KnowledgeSource["kind"], IconSvgElement> = {
  conversation: BubbleChatIcon,
  document: File01Icon,
  person: UserIcon,
  agent: SparklesIcon,
  app: BookOpen01Icon,
};

/** Where it came from: a conversation (a link, when the app has Conversations), a document, a person. */
function SourceNote({ source, conversations }: { source: KnowledgeSource; conversations: string | null }) {
  // A person's or a document's name as it is; "A conversation" and "The Agent" read on from "From".
  const label = `From ${source.kind === "person" || source.kind === "document" ? source.label : lowerFirst(source.label)}`;
  const content = (
    <>
      <HugeiconsIcon icon={SOURCE_ICONS[source.kind]} strokeWidth={2} className="size-3.5" aria-hidden="true" />
      {label}
    </>
  );
  if (source.conversation && conversations) {
    return (
      <Link
        to={`${conversations}/${encodeURIComponent(source.conversation)}`}
        className="inline-flex items-center gap-1.5 underline-offset-4 hover:text-foreground hover:underline"
      >
        {content}
      </Link>
    );
  }
  return <span className="inline-flex items-center gap-1.5">{content}</span>;
}

function Moment({ verb, at }: { verb: string; at: string }) {
  return (
    <time dateTime={at} title={formatDateTime(at)}>
      {verb} {formatRelative(at)}
    </time>
  );
}

function GroupBadge({ group }: { group: string }) {
  return (
    <Badge variant="outline" className="ml-1 font-normal">
      {group}
    </Badge>
  );
}

function Meta({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">{children}</div>;
}

function Dot() {
  return <span aria-hidden="true">·</span>;
}

const lowerFirst = (text: string) => (text ? text.charAt(0).toLowerCase() + text.slice(1) : text);

function NothingYet() {
  return (
    <Empty className="border border-dashed">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <HugeiconsIcon icon={BookOpen01Icon} strokeWidth={2} />
        </EmptyMedia>
        <EmptyTitle>The Agent hasn't kept anything yet</EmptyTitle>
        <EmptyDescription>
          When it learns something worth remembering in its work, it suggests it. Once someone keeps a suggestion, the Agent uses it, and it shows here.
        </EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}

function NoMatch({ q, onClear }: { q: string; onClear: () => void }) {
  return (
    <Empty className="border border-dashed">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <HugeiconsIcon icon={Search01Icon} strokeWidth={2} />
        </EmptyMedia>
        <EmptyTitle>Nothing matches “{q}”</EmptyTitle>
        <EmptyDescription>Try other words. The search looks for whole words, and their other forms.</EmptyDescription>
      </EmptyHeader>
      <EmptyContent>
        <Button variant="outline" size="sm" onClick={onClear}>
          Clear the search
        </Button>
      </EmptyContent>
    </Empty>
  );
}

function StatementsSkeleton() {
  return (
    <Card className="py-0">
      <CardContent className="flex flex-col gap-5 py-5">
        {[0, 1, 2].map((row) => (
          <div key={row} className="flex flex-col gap-2">
            <Skeleton className="h-4 w-4/5" />
            <Skeleton className="h-3 w-1/3" />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
