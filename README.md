# dsh-compaction-tools-gate

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that
stops automatic compaction from **silently losing its summary**.

Mitigates [discussion #5521](https://github.com/deepseek-ai/deepseek-harness/discussions/5521),
confirmed by a maintainer as *"a real, reproducible data loss defect"*. **This
plugin is a stopgap.** The correct fix is upstream, and when it lands you
should delete this.

## The defect

`compaction-basic`'s summarizer replays the conversation's tool schemas into
the summarization call — deliberately, to reuse the warm prefix cache
(`summarizer.ts`, comments 111–114):

```ts
...input.tools === undefined ? {} : { tools: [...input.tools] },   // summarizer.ts:158
```

But `GenerateOptions` (`llm/src/types.ts:393`) has **no `toolChoice` field** —
no way to say *"you have tools, don't call one."* Its own doc comment says so:

> `GenerateOptions` sampling is `temperature`/`maxTokens`/`stop` only — no
> `tool_choice`, `top_p`, or penalty fields

So a tool-capable model, deep in a tool-calling session, can answer the
"you are now acting as a compaction engine" instruction **with another tool
call**. `summaryText()` keeps text blocks only, finds none, and throws
`summarization produced no text summary content` (`summarizer.ts:171`). It
retries until the provider returns 400.

Nothing in the session log distinguishes that from a slow compaction. The
`rawOutput` is discarded before the throw, so one generic string covers three
different bugs: a genuinely empty completion, a truncation, and a model that
chose the wrong behaviour.

## What this does

Registers one listener on the `llm/stream` waterfall and, for calls tagged
`purpose: 'compaction'` **only**, removes the `tools` array.

```yaml
- insert:
    - id: compaction-tools-gate
      name: 'dsh-compaction-tools-gate'
```

Two things make the seam the right place rather than a config change:

- **It reaches the acting instance.** Under `dsh web` the app disables the
  host-plane `compaction-basic` and the `standard` preset mounts its own inside
  an isolated group — so a home-patch config row configures a *disabled*
  instance. Under `--profile headless` the host-plane one is live. The
  `llm/stream` waterfall reaches both.
- **No preset fork, no patched vendor code.** The plugin has **zero
  dependencies** and imports nothing from `@deepseek-ai`.

## The cost, measured

The tools are replayed for cache reuse, so dropping them costs a re-prefill.
On one box — `qwen3.8-27b` (Q5_K_XL) on llama.cpp across 2× RTX A4000, 131k
window, one slot, ~815 t/s prefill:

**Over 74 compactions in real sessions:**

| | |
|---|---|
| compactions that produced no summary | **25 of 64 (39%)** |
| of those, this defect's signature | 15 (12 in a single session) |
| trigger point, median input | 65k tokens of a 131k window |
| total cost of one compaction | **~208 s**, once per ~54k tokens of growth |
| — summarizing | 131 s (decode-bound, ~22 t/s) |
| — re-prefill caused by dropping the tools | **77 s** |

**From one captured compaction** (a single sample, not a distribution):
87% of the summarization request was tool *results*, not schemas or assistant
text; capping them made the request 43% smaller and the re-prefill 88 s → 50 s.

So the trade is roughly **77 s per compaction against a 39% chance of losing
the summary entirely**. On this hardware that is not a close call. Your
mileage depends on prefill speed: at 815 t/s the re-prefill is a minute; on a
faster lane it is seconds, and on a slower one you should think harder.

## Options

```yaml
config:
  # Cap oversized tool results inside the summarized region. 0 disables.
  # The trade is real: a summary can miss what only lived in a truncated tail.
  # Every cut is marked in-place; assistant text is never touched.
  toolResultMaxChars: 2000

  # Send compaction to a different model. A THINKING lane spends the
  # summarizer's token budget on reasoning and truncates; its nothink sibling
  # of the same weights writes the summary with the whole budget, faster.
  reroute:
    'provider/thinking-model': 'nothink-model'

  quiet: true   # no log line per gated call
```

## Scope, and what it does NOT fix

The gate keys on `options.purpose !== 'compaction'` and passes everything else
through. That is deliberate — a `llm/stream` listener sees every request, and a
broad tool-stripping rule there is a footgun.

It is also why this **cannot** substitute for the upstream fix: any *other*
synthetic call that carries tools and must not use one has the same exposure.
A general optional `toolChoice` on `GenerateOptions` is the right shape, and if
it lands compaction-specific this plugin is still needed for the next call of
that class.

## Install

```
npm i dsh-compaction-tools-gate
```

Then add the `insert` row above to your `cordis.patch.yml`, or apply the
bundled patch via the `dsh.bundle` field.

## Test

```
npm test        # 36 checks, no network, no model
```

Verified against dsh **0.1.2-alpha.5** and **0.1.2-rc.1**; the defect is
present and unchanged in both.

## License

MIT
