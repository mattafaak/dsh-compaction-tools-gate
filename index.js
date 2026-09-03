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
 * NO DEPENDENCIES, loaded by absolute file:// URL from a cordis.patch.yml
 * row, for the same reason as dsh-web-search-searxng: nothing here may
 * require a pnpm install, because any pnpm run restores the web-auth prompt.
 */

/** Plugin name shown by plugin-listing surfaces. */
export const name = 'llm-compaction-shim'

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
export function classify (options, reroute = {}, toolResultMaxChars = 0) {
  if (!options || options.purpose !== COMPACTION_PURPOSE) return { action: 'pass' }
  const dropTools = Array.isArray(options.tools) && options.tools.length > 0
  const target = reroute[`${options.provider}/${options.model}`]
  const model = typeof target === 'string' && target.length > 0 && target !== options.model ? target : undefined
  const trim = toolResultMaxChars > 0 && countOversized(options.messages, toolResultMaxChars) > 0
  if (!dropTools && model === undefined && !trim) return { action: 'pass' }
  return { action: Object.isFrozen(options) ? 'redispatch' : 'mutate', dropTools, model, trim }
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
  // 0 disables the trim. 2000 is Yunado's tested default (#3465); it is not
  // the default HERE, because a config that trims by default would change what
  // summaries are built from without anyone choosing it.
  const maxChars = Number.isFinite(config.toolResultMaxChars) ? config.toolResultMaxChars : 0
  const log = (msg) => { if (!quiet) console.error(`[llm-compaction-shim] ${msg}`) }
  ctx.on('llm/stream', function (options, next) {
    const v = classify(options, reroute, maxChars)
    if (v.action === 'pass') return next()
    const parts = []
    if (v.dropTools) parts.push(`dropping ${options.tools.length} tool schemas so the summarizer answers in text`)
    if (v.model) parts.push(`rerouting to ${options.provider}/${v.model} (no reasoning in the summary budget)`)
    if (v.trim) parts.push(`capping ${countOversized(options.messages, maxChars)} tool result(s) at ${maxChars} chars`)
    log(`compaction call on ${options.provider}/${options.model}: ${parts.join('; ')} (maxTokens ${options.maxTokens})`)
    if (v.action === 'mutate') {
      // The waterfall's inner callback closes over this same object, so the
      // adapter sees the change; a replacement object would not reach it.
      if (v.dropTools) delete options.tools
      if (v.model) options.model = v.model
      if (v.trim) options.messages = trimToolResults(options.messages, maxChars)
      return next()
    }
    // Frozen options: re-enter the waterfall with a copy. `this` is the llm
    // runtime the event is bound to; the copy has no tools and the target
    // model, so this listener passes it straight through on the second dispatch.
    if (typeof this?.stream === 'function') {
      const { tools: _dropped, ...rest } = options
      if (v.model) rest.model = v.model
      if (v.trim) rest.messages = trimToolResults(rest.messages, maxChars)
      return this.stream(rest)
    }
    log('options are frozen and no runtime handle is bound; passing through unchanged')
    return next()
  }, { global: true })
}

export default { name, inject, apply }
