import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// An agent's reply as Markdown. Raw HTML in it is shown as text, never run. Unsafe link
// targets are dropped, and images are shown as their description: loading one would send a
// request wherever the reply points, without anyone clicking.
const components: Components = {
  p: ({ node: _, ...props }) => <p className="leading-relaxed" {...props} />,
  a: ({ node: _, href, children, ...props }) =>
    href ? (
      <a
        className="text-primary underline underline-offset-4"
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        {...props}
      >
        {children}
      </a>
    ) : (
      <span>{children}</span>
    ),
  img: ({ alt }) => <span className="text-muted-foreground">[{alt || "image"}]</span>,
  ul: ({ node: _, ...props }) => <ul className="list-disc space-y-1 pl-5" {...props} />,
  ol: ({ node: _, ...props }) => <ol className="list-decimal space-y-1 pl-5" {...props} />,
  h1: ({ node: _, ...props }) => <h3 className="text-base font-semibold" {...props} />,
  h2: ({ node: _, ...props }) => <h3 className="text-base font-semibold" {...props} />,
  h3: ({ node: _, ...props }) => <h4 className="font-semibold" {...props} />,
  blockquote: ({ node: _, ...props }) => (
    <blockquote className="border-l-2 border-border pl-3 text-muted-foreground" {...props} />
  ),
  code: ({ node: _, className, ...props }) => (
    <code
      className={
        className
          ? `${className} font-mono text-sm`
          : "rounded bg-muted px-1 py-0.5 font-mono text-sm"
      }
      {...props}
    />
  ),
  pre: ({ node: _, ...props }) => (
    <pre
      className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-sm [&>code]:bg-transparent [&>code]:p-0"
      {...props}
    />
  ),
  table: ({ node: _, ...props }) => (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm" {...props} />
    </div>
  ),
  th: ({ node: _, ...props }) => (
    <th className="border-b border-border px-2 py-1 text-left font-medium" {...props} />
  ),
  td: ({ node: _, ...props }) => <td className="border-b border-border px-2 py-1" {...props} />,
  hr: ({ node: _, ...props }) => <hr className="border-border" {...props} />,
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
