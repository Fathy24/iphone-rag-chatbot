import { useState } from "react";
import {
  Gauge,
  Loader2,
  Brain,
  Layers,
  Sparkles,
  BarChart3,
  SlidersHorizontal,
  Map,
  Quote,
  ShieldCheck,
  ShieldAlert,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
} from "lucide-react";

// Single, self-contained "live session dashboard" element. It merges the
// context-window meter and the session panel into ONE custom element so the
// whole thing can be relocated into a slide-in side panel by public/custom.js
// (the root carries a stable `data-cl-dashboard` anchor). Everything is
// prop-driven: props.context feeds the meter, props.panel feeds the panel.

function fmt(n) {
  if (n == null) return "0";
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "K";
  return String(n);
}

function clampPct(p) {
  if (p == null || isNaN(p)) return 0;
  return Math.max(0, Math.min(100, p));
}

// --- shared primitives -------------------------------------------------------

function Section({ icon: Icon, title, badge, defaultOpen, children }) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div className="rounded-lg border border-border bg-muted/20">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-2.5 py-2 text-left text-xs font-semibold text-muted-foreground hover:bg-muted/40"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-70" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 opacity-70" />
        )}
        <Icon className="h-3.5 w-3.5 shrink-0" />
        <span className="flex-1">{title}</span>
        {badge != null && (
          <span className="rounded-full bg-muted px-1.5 py-0.5 font-mono text-[10px] text-foreground">
            {badge}
          </span>
        )}
      </button>
      {open && <div className="px-2.5 pb-2.5 pt-0.5">{children}</div>}
    </div>
  );
}

function Row({ label, value, mono = true }) {
  return (
    <div className="flex items-center justify-between gap-2 py-0.5 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className={mono ? "font-mono" : ""}>{value}</span>
    </div>
  );
}

function Chip({ children, tone = "muted" }) {
  const tones = {
    muted: "bg-muted/60 text-foreground",
    on: "bg-primary/15 text-primary border border-primary/30",
    off: "bg-muted/40 text-muted-foreground",
  };
  return (
    <span className={"rounded-md px-1.5 py-0.5 font-mono text-[10px] " + (tones[tone] || tones.muted)}>
      {children}
    </span>
  );
}

// --- context-window meter ----------------------------------------------------

function ContextCard({ c }) {
  const running = !!c.running;
  const pct = clampPct(c.pct);
  const peak = c.peakTokens || 0;
  const budget = c.budget || 0;
  const turn = c.turnTokens || {};
  const convoTurns = c.convoTurns || 0;
  const foldThreshold = c.foldThreshold || 0;
  const foldedTurns = c.foldedTurns || 0;
  const summaryEnabled = c.summaryEnabled !== false;
  const summaryActive = !!c.summaryActive;
  const justSummarized = !!c.justSummarized;

  const barColor =
    pct >= 85 ? "bg-red-500" : pct >= 60 ? "bg-amber-500" : "bg-primary";
  const memPct = foldThreshold
    ? clampPct((Math.min(convoTurns, foldThreshold) / foldThreshold) * 100)
    : 0;

  const remaining = Math.max(0, foldThreshold - convoTurns);
  let summaryLine;
  if (!summaryEnabled) {
    summaryLine = "Rolling summary disabled";
  } else if (summaryActive) {
    summaryLine = `Rolling summary active · ${foldedTurns} message${foldedTurns === 1 ? "" : "s"} folded`;
  } else if (remaining > 0) {
    summaryLine = `Summarizes in ${remaining} more message${remaining === 1 ? "" : "s"} (at ${foldThreshold})`;
  } else {
    summaryLine = "Summarizing older messages now…";
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-muted/20 p-2.5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
          {running ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Gauge className="h-3.5 w-3.5" />
          )}
          Context window
        </div>
        <div className="font-mono text-xs text-muted-foreground">
          ~{fmt(peak)} / {fmt(budget)} tok · {pct.toFixed(0)}%
        </div>
      </div>

      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={"h-full rounded-full transition-all duration-500 " + barColor}
          style={{ width: pct + "%" }}
        />
      </div>
      <div className="text-[10px] text-muted-foreground">
        Largest prompt this session vs the model's {fmt(budget)}-token limit.
      </div>

      <div className="flex flex-col gap-1 border-t border-border pt-2">
        <div className="flex items-center gap-1.5 text-xs">
          <Layers className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="text-muted-foreground">Memory window</span>
          <span className="font-mono" title="Messages kept verbatim before older ones fold into the summary">
            {convoTurns}/{foldThreshold} msgs
          </span>
          <div className="ml-1 h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary/60 transition-all duration-500"
              style={{ width: memPct + "%" }}
            />
          </div>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <Brain className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className={summaryActive ? "text-foreground" : "text-muted-foreground"}>
            {summaryLine}
          </span>
        </div>
      </div>

      {justSummarized && (
        <div className="flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2 py-1 text-xs text-primary">
          <Sparkles className="h-3.5 w-3.5 animate-pulse" />
          Older turns just folded into the rolling summary
        </div>
      )}

      {(turn.total || 0) > 0 && (
        <div className="font-mono text-[11px] text-muted-foreground">
          last turn: {fmt(turn.total)} tok ({fmt(turn.input)} in / {fmt(turn.output)} out)
          {turn.calls ? ` · ${turn.calls} LLM call${turn.calls === 1 ? "" : "s"}` : ""}
        </div>
      )}
    </div>
  );
}

// --- session panel -----------------------------------------------------------

function PanelCards({ p }) {
  const stats = p.stats || {};
  const config = p.config || {};
  const summary = p.summary || {};
  const guideMap = p.guideMap || {};
  const citations = p.citations || {};
  const guard = p.guard || {};

  const sections = guideMap.sections || [];
  const maxCites = Math.max(1, ...sections.map((s) => s.cites || 0));
  const citeItems = citations.items || [];

  return (
    <>
      <Section icon={BarChart3} title="Session" badge={`${stats.turns || 0} turns`} defaultOpen>
        <Row label="Tokens" value={`${fmt(stats.total)} (${fmt(stats.input)} in / ${fmt(stats.output)} out)`} />
        <Row label="LLM calls" value={fmt(stats.calls)} />
        <Row label="Avg latency" value={`${(stats.avgLatency || 0).toFixed(1)}s`} />
        <Row label="Est. cost" value={`≈ $${(stats.cost || 0).toFixed(4)}`} />
        <div className="pt-1 text-[10px] text-muted-foreground">
          Cost at public gpt-4o rates — indicative only.
        </div>
      </Section>

      <Section icon={SlidersHorizontal} title="Retrieval">
        <div className="flex flex-wrap gap-1 pb-1.5">
          <Chip>{config.backend || "?"}</Chip>
          <Chip>{config.mode || "?"}</Chip>
          <Chip tone={config.reranker ? "on" : "off"}>
            rerank {config.reranker ? "on" : "off"}
          </Chip>
        </div>
        <Row label="Coarse sections" value={`${config.coarseN} / ${config.sectionsTotal || "?"}`} />
        <Row label="Fine top-k" value={config.topK} />
        <Row label="Chat" value={config.chatModel} mono={false} />
        <Row label="Embed" value={config.embedModel} mono={false} />
        <div className="pt-1 text-[10px] text-muted-foreground">
          Toggle the reranker / sliders in the ⚙️ settings panel.
        </div>
      </Section>

      <Section
        icon={Brain}
        title="Rolling summary"
        badge={summary.active ? `${summary.foldedTurns} folded` : "off"}
      >
        {summary.active ? (
          <div className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded-md bg-muted/40 p-2 text-xs leading-relaxed text-foreground/90">
            {summary.text}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            No summary yet — older turns fold in once the conversation grows.
          </div>
        )}
      </Section>

      <Section icon={Map} title="Guide map" badge={`${sections.length}`}>
        <div className="flex max-h-56 flex-col gap-0.5 overflow-y-auto">
          {sections.map((s, i) => (
            <div
              key={i}
              className={
                "flex items-center gap-1.5 rounded px-1.5 py-1 text-xs " +
                (s.hit ? "bg-primary/10" : "")
              }
            >
              <span
                className={
                  "h-1.5 w-1.5 shrink-0 rounded-full " +
                  (s.hit ? "bg-primary" : "bg-muted-foreground/30")
                }
                title={s.hit ? "Searched this turn" : ""}
              />
              <span className="flex-1 truncate" title={s.title}>
                {s.title}
              </span>
              {s.pages && <span className="font-mono text-[10px] text-muted-foreground">{s.pages}</span>}
              {s.cites > 0 && (
                <span
                  className="rounded-full px-1.5 py-0.5 font-mono text-[10px] text-primary"
                  style={{ backgroundColor: `rgba(236,72,153,${0.12 + 0.5 * (s.cites / maxCites)})` }}
                  title={`${s.cites} citation${s.cites === 1 ? "" : "s"} this session`}
                >
                  {s.cites}
                </span>
              )}
            </div>
          ))}
          {sections.length === 0 && (
            <div className="text-xs text-muted-foreground">Guide map unavailable.</div>
          )}
        </div>
        <div className="pt-1 text-[10px] text-muted-foreground">
          Dot = searched this turn · number = citations this session.
        </div>
      </Section>

      <Section icon={Quote} title="Citations" badge={`${citations.total || 0}`}>
        {citeItems.length ? (
          <div className="flex max-h-44 flex-col gap-0.5 overflow-y-auto">
            {citeItems.map((c, i) => (
              <div key={i} className="flex items-center gap-1.5 py-0.5 text-xs">
                <span className="font-mono text-muted-foreground">p.{c.page}</span>
                <span className="flex-1 truncate" title={c.section}>
                  {c.section}
                </span>
                {c.count > 1 && <span className="font-mono text-[10px] text-muted-foreground">×{c.count}</span>}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">No citations yet.</div>
        )}
      </Section>

      <Section
        icon={guard.lastBlocked ? ShieldAlert : ShieldCheck}
        title="Guardrail"
        badge={guard.blockedCount ? `${guard.blockedCount} blocked` : "clean"}
      >
        <div className="flex items-center gap-1.5 text-xs">
          {guard.lastBlocked ? (
            <>
              <XCircle className="h-3.5 w-3.5 text-red-500" />
              <span>
                Last input blocked
                {guard.lastReason ? ` — ${guard.lastReason}` : ""}
              </span>
            </>
          ) : (
            <>
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
              <span className="text-muted-foreground">Last input passed safety checks</span>
            </>
          )}
        </div>
        <Row label="Blocked this session" value={guard.blockedCount || 0} />
      </Section>
    </>
  );
}

export default function Dashboard() {
  const c = props.context || {};
  const p = props.panel || {};
  return (
    <div data-cl-dashboard className="flex flex-col gap-2">
      <ContextCard c={c} />
      <PanelCards p={p} />
    </div>
  );
}
