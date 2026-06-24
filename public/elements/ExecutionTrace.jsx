import { useState } from "react";
import { ChevronRight, ChevronDown, Loader2, Activity } from "lucide-react";

// Minimal inline Markdown -> React (bold + `code`) so step bodies keep their
// emphasis without pulling in a Markdown dependency.
function renderInline(text, keyBase) {
  const parts = [];
  const regex = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let idx = 0;
  let m;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      parts.push(<strong key={`${keyBase}-${idx}`}>{tok.slice(2, -2)}</strong>);
    } else {
      parts.push(
        <code
          key={`${keyBase}-${idx}`}
          className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]"
        >
          {tok.slice(1, -1)}
        </code>
      );
    }
    last = m.index + tok.length;
    idx += 1;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function renderBody(body) {
  return (body || "").split("\n").map((line, i) => {
    if (!line.trim()) return <div key={i} className="h-1.5" />;
    return (
      <div key={i} className="py-0.5 leading-relaxed">
        {renderInline(line, `l${i}`)}
      </div>
    );
  });
}

export default function ExecutionTrace() {
  const steps = props.steps || [];
  const running = !!props.running;
  // Every step starts collapsed; the user expands what they want to inspect.
  const [open, setOpen] = useState({});

  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-border bg-muted/20 p-2">
      <div className="flex items-center gap-1.5 px-1 pb-1 text-xs font-semibold text-muted-foreground">
        {running ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <Activity className="h-3.5 w-3.5" />
        )}
        Execution trace
        <span className="font-normal">
          · {steps.length} step{steps.length === 1 ? "" : "s"}
          {running ? " · running…" : ""}
        </span>
      </div>

      {steps.map((s, i) => {
        const isOpen = !!open[i];
        const hasBody = !!(s.body && s.body.trim());
        return (
          <div key={i} className="rounded-md">
            <button
              onClick={() => setOpen((o) => ({ ...o, [i]: !o[i] }))}
              className="flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-sm hover:bg-muted"
            >
              {hasBody ? (
                isOpen ? (
                  <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-70" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5 shrink-0 opacity-70" />
                )
              ) : (
                <span className="inline-block h-3.5 w-3.5 shrink-0" />
              )}
              <span className="font-medium">{s.title}</span>
            </button>
            {isOpen && hasBody && (
              <div className="ml-[14px] border-l border-border pb-2 pl-3 pt-0.5 text-sm text-muted-foreground">
                {renderBody(s.body)}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
