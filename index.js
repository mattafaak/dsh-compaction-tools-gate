/**
 * dsh-llm-compaction-shim -- make the compaction summarizer answer in TEXT on a
 * tool-capable local lane.
 *
 * THE DEFECT THIS CLOSES (measured on alder, 2026-09-02)
 * =====================================================
 * dsh-compaction-basic builds its summarization request as a genuine prefix of
 * the conversation -- system prompt, the 28 tool schemas, the region being
 * condensed -- and appends "You are now acting as a compaction engine ..." as
 * the final user message, so the provider's KV cache is reused. It sets no
 * tool choice, and the pi-ai adapter forwards none. A tool-capable model that
 * has spent the whole session emitting tool calls answers that instruction
 * with another tool call; `summaryText()` keeps only text blocks, so the
 * plugin throws "summarization produced no text summary content", retries at
 * every step, and the context grows until the lane returns 400.
 *
 * Counted over 74 compactions in ~/.dsh/sessions: 15 of the 25 failures were
 * exactly this, 12 of them in one session on qwen3.8-27b-vl, 3-17 s each --
 * the duration of a tool call, not of a summary. The first captured
 * summarization request (dsh-capture record 0131) carried `tools: [28]` and
 * no `tool_choice`.
 *
 * WHY THIS SHAPE
 * ==============
 * `toolChoice: "none"` would be the free fix (the prompt would stay a cache
 * prefix), but nothing host-patchable can send it: dsh-llm's GenerateOptions
 * has no such field and `dsh-llm-pi-ai` assembles the pi-ai options
 * explicitly (temperature, maxTokens, sessionId, signal, headers). Dropping
 * `tools` is what CAN be done from the `llm/stream` waterfall, and it makes
 * the text reply certain rather than likely. The price is one re-prefill of
 * the condensed region per compaction (the tool schemas sit inside the
 * system region of Qwen's template, so removing them invalidates the prefix):
 * ~64k tokens at the ~510 t/s measured at that depth, about two minutes,
 * against a failure mode that ends the session.
 *
 * SECOND JOB (same seam): a THINKING lane's summarization call is rerouted to
 * its nothink sibling -- see `classify()` and the `reroute` config.
 *
 * THIRD JOB: cap oversized tool results in the region being summarized.
 * Measured on a real compaction here (capture 0285): 87% of the 179,471-char
 * request was tool results, 29 of them over 2,000 chars. Capping at 2,000 makes
 * the request 43% smaller -- 44,867 -> 25,709 tokens, which at the measured
 * 510 t/s at depth is 88 s -> 50 s of prefill, EVERY compaction. That prefill
 * is unavoidable once the tool schemas are dropped (the cache prefix is gone
 * either way), so a smaller region is a directly shorter one.
 *
 * The idea and the 2,000 default are Yunado's, from
 * deepseek-ai/deepseek-harness#3465 and the dsh-qwen38-local-qol plugin, tested
 * there across 128k/150k/256k sessions. The plugin itself was NOT adopted here:
 * it reaches the compaction row by FORKING the standard preset into
 * ~/.dsh/.agent-presets (a standing no on this box), installs through pnpm
 * (which restores two vendored patches), and brings its own provider route that
 * would replace four tuned lanes. This seam needs none of that.
 *
 * THE TRADE, stated plainly: a summary built on truncated tool results can miss
 * something that only existed in the truncated tail. Each cut carries a marker
 * so the model knows it is reading a fragment, the assistant's own text is
 * never touched, and `toolResultMaxChars: 0` disables it.
 *
 * FOURTH JOB (2026-09-05): KEEP THE CACHE PREFIX. Dropping the schemas made
 * every summarize call a cold prefill -- 10 of 10 on 2026-09-04 reused ZERO
 * cache, 105-134 s each -- because the schemas sit in Qwen's system region.
 * llama.cpp keeps the tools in the rendered template under `tool_choice: none`
 * and only skips the grammar (common/chat.cpp `include_grammar`, verified at
 * b10797 and b10816), and llama-swap v253's `filters.setParamsByID` pins that
 * parameter onto an ALIAS of the same process. So with `keepToolsVia:
 * {"alder/qwen3.8-27b-vl": "qwen3.8-27b-vl-compact-notools"}` a compaction
 * call goes to the alias WITH its tools and WITHOUT the trim (the trim
 * rewrites the middle and would kill the prefix). Measured by effect before
 * shipping: the alias request reused 656 of 660 tokens from the base lane's
 * request, and returned `finish: stop` where the base lane returned a tool
 * call. THE PRICE: under tool_choice none the model can still write the tool
 * call AS TEXT (`<tool_call> <function=...`) -- it did, under a tool-forcing
 * prompt -- and a summary that is a tool call is a silent data loss. So the
 * alias reply is BUFFERED (compaction is not interactive), inspected at its
 * end, and a tool-shaped, empty or errored reply is replaced by the certain
 * cold path (tools dropped, results trimmed), counted and logged. The
 * thinking lane's reroute keeps the cold path: a lane swap is cold anyway.
 *
 * FIFTH JOB (2026-09-28): the SESSION TITLE call on a thinking lane goes to
 * that lane's thinking-off alias -- see `titleTarget()` and `titleReroute`.
 * dsh-session-title-llm sends 64 max tokens and keeps text blocks only; on
 * spark/qwen3.8-flash-next all 64 went to reasoning (finish=length, content
 * null), so 0 of 115 sessions on that lane ever got a model title. The alias
 * is the SAME process with enable_thinking false (Spark llama-swap
 * setParamsByID), so nothing is loaded or evicted. Keyed per lane, like
 * `reroute`, and deliberately NOT the plugin's own provider/model override:
 * that override is global, so a session on another lane of the Spark's
 * exclusive group would have its lane evicted by its own title call.
 *
 * SIXTH JOB (2026-09-29): raise a lane's SUMMARY CAP -- see `summaryMaxTokens`.
 * dsh-compaction-basic asks for 8,192 tokens and fails the compaction on a
 * max-tokens finish ("summarization truncated at the token cap"); it checks
 * nothing else about length. Each summary carries the previous one forward,
 * so over a long session it grows: racr 09-28/29 went 4,208 -> 8,051 output
 * tokens over 12 successful compactions, thinking off, and 13 attempts
 * truncated (one burst of 8 in a row, ~3.5 min each). On 0.1.6 the cap is a
 * preset-owned row, so this seam is the only one that reaches it. Per lane.
 *
 * NO DEPENDENCIES, loaded by absolute file:// URL from a cordis.patch.yml
 * row, for the same reason as dsh-web-search-searxng: nothing here may
 * require a pnpm install, because any pnpm run restores the web-auth prompt.
 */

/** Plugin name shown by plugin-listing surfaces. */
// The package identity, so a user sees ONE name: the bundle patch mounts
// `id: compaction-tools-gate` and cordis logs the same string. Before 0.1.1
// this said 'llm-compaction-shim' -- the local file's original name -- so the
// config said one thing and the log another.
export const name = 'compaction-tools-gate'

/** The waterfall is dispatched by the llm runtime; make sure it exists first. */
export const inject = ['llm']

/** The purpose tag dsh-compaction-basic stamps on its summarization call. */
export const COMPACTION_PURPOSE = 'compaction'

/**
 * Decide what to do with one llm/stream call.
 * @param options - the GenerateOptions of the call.
 * @param reroute - `{ "provider/model": "model" }`: summarization calls on the
 *   key lane go to the value lane (same provider). Reason: a THINKING lane
 *   spends the summarizer's 8,192-token budget on reasoning -- measured
 *   2026-09-02, one compaction on qwen3.8-27b ran 397 s and ended
 *   "truncated at the token cap" -- while the nothink sibling of the same
 *   weights writes the summary at twice the speed with the whole budget. The
 *   region is re-prefilled either way once the tools are gone, so the
 *   marginal cost of the reroute is two lane loads (~20 s), not a prefill.
 * @returns `{ action: 'pass'|'mutate'|'redispatch', dropTools, model }`;
 *   pure, so the test can pin it.
 */
export function classify (options, reroute = {}, toolResultMaxChars = 0, keepToolsVia = {}, summaryMaxTokens = {}) {
  if (!options || options.purpose !== COMPACTION_PURPOSE) return { action: 'pass' }
  const cap = summaryMaxTokens[`${options.provider}/${options.model}`]
  const maxTokens = Number.isInteger(cap) && cap > (options.maxTokens ?? 0) ? cap : undefined
  const dropTools = Array.isArray(options.tools) && options.tools.length > 0
  const target = reroute[`${options.provider}/${options.model}`]
  const model = typeof target === 'string' && target.length > 0 && target !== options.model ? target : undefined
  // KEEP: this lane has a tool_choice-none alias, the call carries tools, and
  // it is not being rerouted to another process (that is cold regardless).
  const via = keepToolsVia[`${options.provider}/${options.model}`]
  if (dropTools && model === undefined && typeof via === 'string' && via.length > 0 && via !== options.model) {
    return { action: 'keep', via, maxTokens }
  }
  const trim = toolResultMaxChars > 0 && countOversized(options.messages, toolResultMaxChars) > 0
  if (!dropTools && model === undefined && !trim && maxTokens === undefined) return { action: 'pass' }
  return { action: Object.isFrozen(options) ? 'redispatch' : 'mutate', dropTools, model, trim, maxTokens }
}

/** The purpose tag dsh-session-title-llm stamps on its title call. */
export const TITLE_PURPOSE = 'session-title'

/**
 * The alias a title call on this lane goes to, or undefined (pass).
 * @param titleReroute - `{ "provider/model": "model" }`, same provider.
 */
export function titleTarget (options, titleReroute = {}) {
  if (!options || options.purpose !== TITLE_PURPOSE) return undefined
  const t = titleReroute[`${options.provider}/${options.model}`]
  return typeof t === 'string' && t.length > 0 && t !== options.model ? t : undefined
}

/**
 * NO TOOL HISTORY ON A TOOLLESS CALL (2026-09-26). Deleting `options.tools` is not
 * enough: pi-ai's openai-completions adapter re-adds `params.tools = []` whenever the
 * history holds tool calls ("Anthropic (via LiteLLM/proxy) requires tools param ..."),
 * and vLLM rejects that with 400 "`tools` must not be an empty array". Every
 * compaction on a vLLM lane (Spark lanes 1 and 3) failed -- 36/36 in the specfloor
 * job on lane 3 -- while llama.cpp on alder accepted it. The adapter's onPayload
 * hook cannot be used: the runtime installs its own (measured: e2e stayed red).
 * So the cold-path summary call gets its tool calls and results as plain TEXT: the
 * summarizer sees the same names, arguments and outputs, and the adapter sees no
 * tool history, so it sends no `tools` key at all.
 */
export function flattenToolHistory (messages) {
  if (!Array.isArray(messages)) return messages
  const asText = (content) => {
    if (typeof content === 'string') return content
    if (!Array.isArray(content)) return ''
    return content.map((b) => (typeof b === 'string' ? b : (b && typeof b.text === 'string' ? b.text : ''))).join('')
  }
  const block = (b) => {
    if (b && b.type === 'tool-call') {
      const args = typeof b.arguments === 'string' ? b.arguments : JSON.stringify(b.arguments ?? {})
      return { type: 'text', text: `[tool call ${b.name}(${args})]` }
    }
    if (b && b.type === 'tool-result') {
      return { type: 'text', text: `[tool result${b.isError ? ' (error)' : ''}]\n${asText(b.content)}` }
    }
    return b
  }
  return messages.map((m) => {
    if (!m || typeof m !== 'object') return m
    if (m.role === 'tool') return { role: 'user', content: [{ type: 'text', text: `[tool result]\n${asText(m.content)}` }] }
    if (Array.isArray(m.content) && m.content.some(b => b && (b.type === 'tool-call' || b.type === 'tool-result'))) {
      return { ...m, content: m.content.map(block) }
    }
    return m
  })
}

/** Text that is a tool call in disguise: what a lane writes under tool_choice none when it wanted to call. */
export function looksLikeToolCall (text) {
  const head = (text || '').trimStart().slice(0, 400)
  return /^(<tool_call>|<function=|\{\s*"(name|tool_calls?)"\s*:)/i.test(head) || /<tool_call>/i.test(head)
}

/** Fallbacks taken so far, so a test or a reader can count them. */
export const stats = { keepCalls: 0, fallbacks: 0 }

/**
 * Consume the alias reply, then either replay it or replace it with the cold
 * path. `inner` is the alias stream (async iterable of chunks), `fallback()`
 * returns the cold-path stream. Pure with respect to everything but `stats`.
 */
export async function * guardedSummary (inner, fallback, log = () => {}) {
  const buf = []
  let text = ''
  let toolCall = false
  let failed = null
  let aborted = false
  try {
    for await (const chunk of inner) {
      buf.push(chunk)
      if (!chunk || typeof chunk !== 'object') continue
      if (chunk.type === 'text-delta') text += chunk.text ?? ''
      else if (chunk.type === 'tool-call-delta') toolCall = true
      else if (chunk.type === 'finish') {
        const kind = chunk.reason?.kind
        if (kind === 'error') failed = chunk.reason?.failure?.message ?? chunk.reason?.failure?.code ?? 'error'
        else if (kind === 'aborted') aborted = true
      }
    }
  } catch (err) {
    failed = err?.message ?? String(err)
  }
  if (aborted) { for (const c of buf) yield c; return }
  let why = null
  if (failed) why = `the alias reply failed (${String(failed).slice(0, 120)})`
  else if (toolCall || looksLikeToolCall(text)) why = 'the alias reply is a tool call in disguise'
  else if (!text.trim()) why = 'the alias reply is empty'
  if (!why) { for (const c of buf) yield c; return }
  stats.fallbacks++
  log(`keep-tools reply fell back to the cold path: ${why} (fallbacks so far: ${stats.fallbacks} of ${stats.keepCalls})`)
  for await (const c of fallback()) yield c
}

/** How many tool results in `messages` exceed `max` characters. */
export function countOversized (messages, max) {
  if (!Array.isArray(messages) || !(max > 0)) return 0
  let n = 0
  for (const m of messages) {
    if (!isToolResult(m)) continue
    if (textLength(m.content) > max) n++
  }
  return n
}

function isToolResult (m) {
  if (!m || typeof m !== 'object') return false
  if (m.role === 'tool') return true
  // dsh's own shape: a user message whose content blocks are tool-result blocks
  return Array.isArray(m.content) && m.content.some(b => b && b.type === 'tool-result')
}

function textLength (content) {
  if (typeof content === 'string') return content.length
  if (!Array.isArray(content)) return 0
  let n = 0
  for (const b of content) {
    if (typeof b === 'string') n += b.length
    else if (b && typeof b.text === 'string') n += b.text.length
    else if (b && Array.isArray(b.content)) n += textLength(b.content)
    else if (b && typeof b.content === 'string') n += b.content.length
  }
  return n
}

/** A copy of `messages` with every oversized tool result capped, each cut marked. */
export function trimToolResults (messages, max) {
  if (!Array.isArray(messages) || !(max > 0)) return messages
  const cut = (text) => {
    const dropped = text.length - max
    return text.slice(0, max) + `\n\n[... ${dropped} characters of this tool result were omitted before summarization ...]`
  }
  const walk = (content) => {
    if (typeof content === 'string') return content.length > max ? cut(content) : content
    if (!Array.isArray(content)) return content
    return content.map((b) => {
      if (typeof b === 'string') return b.length > max ? cut(b) : b
      if (b && typeof b.text === 'string' && b.text.length > max) return { ...b, text: cut(b.text) }
      if (b && typeof b.content === 'string' && b.content.length > max) return { ...b, content: cut(b.content) }
      if (b && Array.isArray(b.content)) return { ...b, content: walk(b.content) }
      return b
    })
  }
  return messages.map((m) => (isToolResult(m) && textLength(m.content) > max
    ? { ...m, content: walk(m.content) }
    : m))
}

export function apply (ctx, config = {}) {
  const quiet = config.quiet === true
  const reroute = config.reroute && typeof config.reroute === 'object' ? config.reroute : {}
  const keepToolsVia = config.keepToolsVia && typeof config.keepToolsVia === 'object' ? config.keepToolsVia : {}
  const titleReroute = config.titleReroute && typeof config.titleReroute === 'object' ? config.titleReroute : {}
  // A YAML-quoted number is a string and would silently do nothing: say so, drop it.
  const summaryMaxTokens = {}
  for (const [lane, v] of Object.entries(config.summaryMaxTokens && typeof config.summaryMaxTokens === 'object' ? config.summaryMaxTokens : {})) {
    if (Number.isInteger(v) && v > 0) summaryMaxTokens[lane] = v
    else console.error(`[llm-compaction-shim] summaryMaxTokens.${lane} is ${JSON.stringify(v)}, not a positive integer -- ignored`)
  }
  // 2000 is Yunado's tested default (#3465); it is not the default HERE,
  // because a config that trims by default would change what summaries are
  // built from without anyone choosing it.
  // 0 disables. A YAML-QUOTED "2000" is a STRING, Number.isFinite says false,
  // and the trim was silently off with no signal anywhere -- so say so instead
  // of falling back quietly. (smoke-test asserts the live value is 2000.)
  let maxChars = 0
  if (Number.isFinite(config.toolResultMaxChars)) {
    maxChars = config.toolResultMaxChars
  } else if (config.toolResultMaxChars !== undefined) {
    const coerced = Number(config.toolResultMaxChars)
    console.error(`[llm-compaction-shim] toolResultMaxChars is ${JSON.stringify(config.toolResultMaxChars)}, ` +
      `not a number -- the tool-result trim is OFF. Unquote it in cordis.patch.yml` +
      (Number.isFinite(coerced) ? ` (you probably meant ${coerced}).` : '.'))
  }

  // A reroute is a map, so a CYCLE is expressible: {a/x: y, a/y: x} would
  // recurse through this.stream() until the stack gave out. Nothing rejects it
  // upstream, and the failure would look like a hung compaction.
  for (const [from, to] of Object.entries(reroute)) {
    const [prov] = from.split('/')
    if (reroute[`${prov}/${to}`] !== undefined) {
      console.error(`[llm-compaction-shim] reroute cycle: ${from} -> ${to} -> ` +
        `${reroute[`${prov}/${to}`]}; dropping ${from} rather than recursing`)
      delete reroute[from]
    }
  }
  const log = (msg) => { if (!quiet) console.error(`[llm-compaction-shim] ${msg}`) }
  ctx.on('llm/stream', function (options, next) {
    const tt = titleTarget(options, titleReroute)
    if (tt !== undefined) {
      log(`title call on ${options.provider}/${options.model}: sending to ${options.provider}/${tt} (thinking off, same process; maxTokens ${options.maxTokens})`)
      // The title plugin deep-freezes its options: re-enter with a copy (the
      // alias has no titleReroute key, so the second dispatch passes).
      if (!Object.isFrozen(options)) { options.model = tt; return next() }
      if (typeof this?.stream === 'function') return this.stream({ ...options, model: tt })
      log('title options are frozen and no runtime handle is bound; passing through unchanged')
      return next()
    }
    const v = classify(options, reroute, maxChars, keepToolsVia, summaryMaxTokens)
    if (v.action === 'pass') return next()
    if (v.action === 'keep') {
      // The fallback needs a runtime handle; without one the certain path is the only path.
      if (typeof this?.stream !== 'function') {
        log(`compaction call on ${options.provider}/${options.model}: no runtime handle for a fallback; taking the cold path`)
      } else {
        stats.keepCalls++
        log(`compaction call on ${options.provider}/${options.model}: keeping ${options.tools.length} tool schemas, sending to alias ${v.via} (prefix cache kept; maxTokens ${v.maxTokens ?? options.maxTokens})`)
        const runtime = this
        const original = { ...options, ...(v.maxTokens ? { maxTokens: v.maxTokens } : {}) }
        const cold = () => {
          const { tools: _dropped, ...rest } = original
          if (maxChars > 0) rest.messages = trimToolResults(rest.messages, maxChars)
          // THE THIRD TOOLLESS PATH (2026-09-26): the keep-tools fallback also drops the
          // tools, so it needs the same flattening or vLLM rejects its `tools: []`.
          rest.messages = flattenToolHistory(rest.messages)
          log(`compaction call on ${options.provider}/${options.model}: cold path -- dropping ${original.tools.length} tool schemas` +
              (maxChars > 0 ? `, capping oversized tool results at ${maxChars} chars` : ''))
          return runtime.stream(rest)
        }
        let inner
        if (Object.isFrozen(options)) inner = runtime.stream({ ...options, model: v.via, ...(v.maxTokens ? { maxTokens: v.maxTokens } : {}) })
        else { options.model = v.via; if (v.maxTokens) options.maxTokens = v.maxTokens; inner = next() }
        return guardedSummary(inner, cold, log)
      }
    }
    const w = v.action === 'keep' ? classify(options, reroute, maxChars, {}, summaryMaxTokens) : v
    const parts = []
    if (w.dropTools) parts.push(`dropping ${options.tools.length} tool schemas so the summarizer answers in text`)
    if (w.model) parts.push(`rerouting to ${options.provider}/${w.model} (no reasoning in the summary budget)`)
    if (w.trim) parts.push(`capping ${countOversized(options.messages, maxChars)} tool result(s) at ${maxChars} chars`)
    if (w.maxTokens) parts.push(`raising the summary cap ${options.maxTokens} -> ${w.maxTokens}`)
    log(`compaction call on ${options.provider}/${options.model}: ${parts.join('; ')} (maxTokens ${w.maxTokens ?? options.maxTokens})`)
    if (w.action === 'mutate') {
      // The waterfall's inner callback closes over this same object, so the
      // adapter sees the change; a replacement object would not reach it.
      if (w.dropTools) { delete options.tools; options.messages = flattenToolHistory(options.messages) }
      if (w.model) options.model = w.model
      if (w.trim) options.messages = trimToolResults(options.messages, maxChars)
      if (w.maxTokens) options.maxTokens = w.maxTokens
      return next()
    }
    // Frozen options: re-enter the waterfall with a copy. `this` is the llm
    // runtime the event is bound to; the copy has no tools and the target
    // model, so this listener passes it straight through on the second dispatch.
    if (typeof this?.stream === 'function') {
      const { tools: _dropped, ...rest } = options
      if (w.dropTools) rest.messages = flattenToolHistory(rest.messages)
      if (w.model) rest.model = w.model
      if (w.trim) rest.messages = trimToolResults(rest.messages, maxChars)
      if (w.maxTokens) rest.maxTokens = w.maxTokens
      return this.stream(rest)
    }
    log('options are frozen and no runtime handle is bound; passing through unchanged')
    return next()
  }, { global: true })
}

export default { name, inject, apply }
