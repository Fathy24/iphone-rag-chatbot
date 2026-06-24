import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { toast } from "sonner";
import {
  Copy,
  FileDown,
  FileText,
  X,
  Search,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

export default function AnswerTools() {
  const groups = props.groups || [];
  const hasSources = !!props.hasSources;
  const turnId = props.turnId;

  // null = nothing open. We NEVER auto-open: the panel only appears after a
  // user clicks a source chip, and the X button closes it again.
  const [openIndex, setOpenIndex] = useState(null);
  const [showContext, setShowContext] = useState(false);

  const allItems = groups.flatMap((g) => g.items || []);
  const selected = allItems.find((it) => it.index === openIndex) || null;

  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(props.answer || "");
      toast.success("Answer copied to clipboard");
    } catch (e) {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  };

  const exportPdf = async (kind, index) => {
    toast.info("Preparing your PDF…");
    try {
      const res = await callAction({
        name: "export_pdf",
        payload: { kind, turnId, index: index ?? null },
      });
      if (!res || res.success === false) toast.error("PDF export failed");
    } catch (e) {
      toast.error("PDF export failed");
    }
  };

  const showTopics = groups.filter((g) => (g.topic || "").trim()).length > 1;

  return (
    <div className="mt-3 flex flex-col gap-3">
      {/* Action bar: copy / download the answer (and all sources) */}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" className="h-7 gap-1.5 text-xs" onClick={copyAnswer}>
          <Copy className="h-3.5 w-3.5" /> Copy answer
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="h-7 gap-1.5 text-xs"
          onClick={() => exportPdf("answer")}
        >
          <FileDown className="h-3.5 w-3.5" /> Answer as PDF
        </Button>
        {hasSources && (
          <Button
            size="sm"
            variant="outline"
            className="h-7 gap-1.5 text-xs"
            onClick={() => exportPdf("sources")}
          >
            <FileText className="h-3.5 w-3.5" /> Sources as PDF
          </Button>
        )}
      </div>

      {/* Source chips, grouped by the topic search that surfaced them */}
      {hasSources && (
        <div className="flex flex-col gap-2 border-t border-border pt-2">
          <div className="text-xs font-medium text-muted-foreground">
            Sources <span className="font-normal">(click a passage to view it)</span>
          </div>
          {groups.map((g, gi) => (
            <div key={gi} className="flex flex-col gap-1.5">
              {showTopics && (g.topic || "").trim() && (
                <div className="flex items-center gap-1 text-xs font-medium text-primary">
                  <Search className="h-3 w-3" /> {g.topic}
                </div>
              )}
              <div className="flex flex-wrap gap-1.5">
                {(g.items || []).map((it) => {
                  const active = it.index === openIndex;
                  return (
                    <button
                      key={it.index}
                      onClick={() => {
                        setShowContext(false);
                        setOpenIndex(active ? null : it.index);
                      }}
                      className={
                        "rounded-md border px-2 py-1 text-xs transition-colors " +
                        (active
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border bg-muted/40 text-foreground hover:bg-muted")
                      }
                    >
                      [{it.index}] p.{it.page} · {it.section}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* The passage viewer — only rendered when a chip is selected */}
      {selected && (
        <Card className="border-primary/30">
          <CardContent className="p-3">
            <div className="mb-2 flex items-start justify-between gap-2">
              <div className="text-sm font-semibold">
                Reference [{selected.index}] — p.{selected.page} · {selected.section}
              </div>
              <button
                onClick={() => setOpenIndex(null)}
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                aria-label="Close passage"
                title="Close"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {selected.scoring && (
              <div className="mb-2 flex flex-col gap-1 rounded-md bg-muted/40 p-2 text-xs">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="w-32 shrink-0 text-muted-foreground">Final rank</span>
                  <Badge className="font-mono">#{selected.scoring.finalRank}</Badge>
                  {selected.scoring.rerank != null && (
                    <span className="font-mono">rerank {selected.scoring.rerank}</span>
                  )}
                  <span className="text-muted-foreground">· after Cohere rerank</span>
                </div>
                {(selected.scoring.preRank != null || selected.scoring.rrf != null) && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="w-32 shrink-0 text-muted-foreground">Hybrid fusion</span>
                    {selected.scoring.preRank != null && (
                      <Badge variant="outline" className="font-mono">
                        #{selected.scoring.preRank}
                      </Badge>
                    )}
                    {selected.scoring.rrf != null && (
                      <span className="font-mono">RRF {selected.scoring.rrf}</span>
                    )}
                    <span className="text-muted-foreground">· dense + BM25, before rerank</span>
                  </div>
                )}
                {(selected.scoring.dense != null ||
                  selected.scoring.bm25 != null ||
                  selected.scoring.score != null) && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="w-32 shrink-0 text-muted-foreground">Raw signals</span>
                    {selected.scoring.dense != null && (
                      <span className="font-mono">dense {selected.scoring.dense}</span>
                    )}
                    {selected.scoring.bm25 != null && (
                      <span className="font-mono">· BM25 {selected.scoring.bm25}</span>
                    )}
                    {selected.scoring.score != null && (
                      <span className="font-mono">score {selected.scoring.score}</span>
                    )}
                  </div>
                )}
              </div>
            )}

            <blockquote className="border-l-2 border-primary/40 pl-3 text-sm leading-relaxed text-foreground/90 whitespace-pre-wrap">
              {selected.text}
            </blockquote>

            {selected.parent && (
              <div className="mt-2">
                <button
                  onClick={() => setShowContext((v) => !v)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  {showContext ? (
                    <ChevronDown className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5" />
                  )}
                  Surrounding context (same section)
                </button>
                {showContext && (
                  <div className="mt-1 whitespace-pre-wrap rounded-md bg-muted/40 p-2 text-xs leading-relaxed text-muted-foreground">
                    {selected.parent}
                  </div>
                )}
              </div>
            )}

            <div className="mt-3">
              <Button
                size="sm"
                variant="outline"
                className="h-7 gap-1.5 text-xs"
                onClick={() => exportPdf("chunk", selected.index)}
              >
                <FileDown className="h-3.5 w-3.5" /> Download this passage (PDF)
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
