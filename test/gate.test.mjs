// Unit test for dsh-llm-compaction-shim: run with `node test_compaction_shim.mjs`.
import { apply, classify } from '../index.js'
let failed = 0
const check = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) failed++ }
const tools = [{ name: 'bash' }, { name: 'read' }]

// registration shape
const listeners = {}
apply({ on: (name, fn, opts) => { listeners[name] = { fn, opts } } }, { quiet: true })
check(typeof listeners['llm/stream']?.fn === 'function', 'registers one llm/stream listener')
check(listeners['llm/stream'].opts?.global === true, 'listener is global (survives the runtime\'s context filter)')
const run = (options, self = {}) => {
  let nextCalled = 0
  const out = listeners['llm/stream'].fn.call(self, options, () => { nextCalled++; return 'inner' })
  return { out, nextCalled }
}

// pure classification
check(classify({ purpose: 'compaction', tools }).action === 'mutate', 'compaction + tools -> mutate')
check(classify(Object.freeze({ purpose: 'compaction', tools })).action === 'redispatch', 'frozen compaction + tools -> redispatch')
check(classify({ purpose: 'compaction', tools: [] }).action === 'pass', 'compaction with empty tools -> pass')
check(classify({ purpose: 'compaction' }).action === 'pass', 'compaction without tools -> pass')
check(classify({ purpose: 'session-title', tools }).action === 'pass', 'session-title call -> pass')
check(classify({ tools }).action === 'pass', 'main-loop call (no purpose) -> pass')
check(classify(undefined).action === 'pass', 'undefined options -> pass')

// behaviour: compaction call loses its tools in place and continues
const c = { purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl', tools: [...tools], maxTokens: 8192, messages: [1] }
let r = run(c)
check(r.nextCalled === 1 && r.out === 'inner', 'compaction: next() called once')
check(!('tools' in c), 'compaction: tools key removed from the SAME object')
check(c.messages.length === 1 && c.maxTokens === 8192, 'compaction: nothing else touched')

// behaviour: main-loop call untouched
const m = { provider: 'alder', model: 'x', tools: [...tools], messages: [1] }
r = run(m)
check(r.nextCalled === 1 && m.tools.length === 2, 'main loop: tools kept, next() called')

// behaviour: frozen compaction options re-dispatch through this.stream without tools
const f = Object.freeze({ purpose: 'compaction', provider: 'p', model: 'm', tools: [...tools], maxTokens: 8192 })
let redispatched = null
r = run(f, { stream: (o) => { redispatched = o; return 'redispatched' } })
check(r.nextCalled === 0 && r.out === 'redispatched', 'frozen: re-dispatched instead of next()')
check(redispatched && !('tools' in redispatched) && redispatched.purpose === 'compaction' && redispatched.maxTokens === 8192, 'frozen: copy has no tools, other fields intact')
// second pass of the copy is a plain pass-through (no infinite loop)
r = run(redispatched, { stream: () => { throw new Error('must not re-dispatch again') } })
check(r.nextCalled === 1, 'frozen: the tool-less copy passes straight through')

// reroute: a thinking lane's summarization goes to the nothink sibling
const RR = { 'alder/qwen3.8-27b': 'qwen3.8-27b-vl' }
let rv = classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b', tools }, RR)
check(rv.action === 'mutate' && rv.dropTools === true && rv.model === 'qwen3.8-27b-vl', 'reroute: thinking lane -> vl, tools dropped too')
rv = classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b' }, RR)
check(rv.action === 'mutate' && !rv.dropTools && rv.model === 'qwen3.8-27b-vl', 'reroute applies even when there are no tools to drop')
check(classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl' }, RR).action === 'pass', 'reroute: the target lane itself passes')
check(classify({ purpose: 'compaction', provider: 'squidward', model: 'qwen3.8-27b', tools }, RR).action === 'mutate' && classify({ purpose: 'compaction', provider: 'squidward', model: 'qwen3.8-27b', tools }, RR).model === undefined, 'reroute is keyed by provider/model, not model alone')
check(classify({ provider: 'alder', model: 'qwen3.8-27b', tools }, RR).action === 'pass', 'reroute never touches a main-loop call')
// behaviour with reroute configured
const L2 = {}
apply({ on: (name, fn) => { L2[name] = fn } }, { quiet: true, reroute: RR })
const c2 = { purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b', tools: [...tools], maxTokens: 8192 }
let n2 = 0; L2['llm/stream'].call({}, c2, () => { n2++ })
check(n2 === 1 && c2.model === 'qwen3.8-27b-vl' && !('tools' in c2), 'mutate path: model rewritten in place and tools dropped')
const f2 = Object.freeze({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b', tools: [...tools], maxTokens: 8192 })
let got = null; L2['llm/stream'].call({ stream: (o) => { got = o } }, f2, () => { throw new Error('must re-dispatch') })
check(got && got.model === 'qwen3.8-27b-vl' && !('tools' in got) && got.maxTokens === 8192, 'redispatch path: copy carries the target model, no tools')

// frozen with no runtime handle: pass through unchanged
r = run(Object.freeze({ purpose: 'compaction', tools: [...tools] }), {})
check(r.nextCalled === 1, 'frozen without this.stream: passes through')

// --- tool-result trim (Yunado #3465's idea, our seam) -------------------------
import { countOversized, trimToolResults } from '../index.js'
const bigTool = { role: "tool", content: "x".repeat(5000) }
const blockTool = { role: "user", content: [{ type: "tool-result", text: "y".repeat(4000) }] }
const msgs = [{ role: "system", content: "sys" }, bigTool,
              // OVER the limit on purpose: with a short assistant message this
              // test cannot tell "assistant text is never trimmed" from "it was
              // too small to trim", and a regression that trims assistant text
              // passed it silently (found by red-proofing, 2026-09-03).
              { role: "assistant", content: "A".repeat(6000) },
              { role: "tool", content: "short" }, blockTool]

check(countOversized(msgs, 2000) === 2, "counts both string and block-shaped oversized tool results")
check(countOversized(msgs, 0) === 0, "max 0 counts nothing (the trim is off)")
const trimmed = trimToolResults(msgs, 2000)
check(trimmed[1].content.length > 2000 && trimmed[1].content.length < 2200,
      "an oversized tool result is capped near the limit, plus a marker")
check(trimmed[1].content.includes("omitted before summarization"),
      "every cut carries a marker, so the model knows it is reading a fragment")
check(trimmed[2].content.length === 6000 && !trimmed[2].content.includes("omitted"),
      "an OVERSIZED assistant message is still never touched (only tool results are)")
check(trimmed[3].content === "short", "a small tool result is left alone")
check(trimmed[4].content[0].text.includes("omitted"), "block-shaped tool results are trimmed too")
check(msgs[1].content.length === 5000, "the ORIGINAL messages are not mutated (a copy is returned)")

// classification: the trim alone is enough to act on
check(classify({ purpose: "compaction", provider: "alder", model: "m", messages: msgs }, {}, 2000).trim === true,
      "a compaction call with oversized tool results is acted on even with no tools and no reroute")
check(classify({ purpose: "compaction", provider: "alder", model: "m", messages: msgs }, {}, 0).action === "pass",
      "with the trim off and nothing else to do, the call passes through")
check(classify({ provider: "alder", model: "m", messages: msgs }, {}, 2000).action === "pass",
      "a MAIN-LOOP call is never trimmed, however big its tool results")

// behaviour through the listener
const L3 = {}
apply({ on: (n, fn) => { L3[n] = fn } }, { quiet: true, toolResultMaxChars: 2000 })
const call = { purpose: "compaction", provider: "alder", model: "m", messages: msgs.map(m => ({ ...m })), maxTokens: 8192 }
let n3 = 0; L3["llm/stream"].call({}, call, () => { n3++ })
check(n3 === 1 && call.messages[1].content.length < 2200 && call.messages[2].content.length === 6000,
      "listener: trims in place, leaves assistant text alone")

// --- config hazards, added 2026-09-03 ---------------------------------------
// Both of these are shapes a person can write in cordis.patch.yml that fail
// SILENTLY: a YAML-quoted number, and a reroute map that points back at itself.
{
  const errs = []
  const realErr = console.error
  console.error = (m) => errs.push(String(m))
  const ctx = { on () {} }
  apply(ctx, { toolResultMaxChars: '2000', quiet: true })
  console.error = realErr
  check(errs.some((e) => e.includes('not a number') && e.includes('trim is OFF')),
        'a quoted toolResultMaxChars says so out loud instead of disabling the trim quietly')
  check(errs.some((e) => e.includes('you probably meant 2000')),
        'and names the value that was meant')
}
{
  const errs = []
  const realErr = console.error
  console.error = (m) => errs.push(String(m))
  const cycle = { 'alder/a': 'b', 'alder/b': 'a' }
  apply({ on () {} }, { reroute: cycle, quiet: true })
  console.error = realErr
  check(errs.some((e) => e.includes('reroute cycle')),
        'a reroute cycle is reported, not recursed into')
  check(Object.keys(cycle).length === 1,
        'and one leg is dropped so the surviving reroute still works')
}


// --- keep the cache prefix via a tool_choice-none alias (2026-09-05) ----------
import { guardedSummary, looksLikeToolCall, stats } from '../index.js'
{
  const KV = { 'alder/qwen3.8-27b-vl': 'qwen3.8-27b-vl-compact-notools' }
  const k = classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl', tools }, {}, 2000, KV)
  check(k.action === 'keep' && k.via === 'qwen3.8-27b-vl-compact-notools', 'keep: a compaction call on a lane with an alias keeps its tools and goes to the alias')
  check(classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl' }, {}, 2000, KV).action === 'pass' ||
        classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl' }, {}, 2000, KV).action !== 'keep', 'keep: a call with no tools is not sent to the alias')
  check(classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b', tools }, RR, 2000, KV).action !== 'keep', 'keep: a rerouted (thinking-lane) call takes the cold path, not the alias')
  check(classify({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl-compact-notools', tools }, {}, 0, { 'alder/qwen3.8-27b-vl-compact-notools': 'qwen3.8-27b-vl-compact-notools' }).action !== 'keep', 'keep: an alias mapped to itself is not a loop')
  check(classify({ provider: 'alder', model: 'qwen3.8-27b-vl', tools }, {}, 0, KV).action === 'pass', 'keep: never touches a main-loop call')
  check(looksLikeToolCall('<tool_call> <function=get_weather> <parameter=city> Paris') && looksLikeToolCall('  {"name": "bash", "arguments": {}}') && !looksLikeToolCall('## Primary Request and Intent\n- build the marble run'),
        'looksLikeToolCall: the text form of a tool call is recognised, a summary is not')

  // the guarded stream: a good summary is replayed verbatim
  const summary = [{ type: 'block-start', index: 0 }, { type: 'text-delta', index: 0, text: '## Summary\n- ok' }, { type: 'block-end', index: 0, block: { type: 'text', text: '## Summary\n- ok' } }, { type: 'finish', reason: { kind: 'stop' } }]
  async function * from (arr) { for (const c of arr) yield c }
  const collect = async (it) => { const out = []; for await (const c of it) out.push(c); return out }
  let fallbackCalls = 0
  const fb = () => { fallbackCalls++; return from([{ type: 'text-delta', index: 0, text: 'COLD SUMMARY' }, { type: 'finish', reason: { kind: 'stop' } }]) }
  let out = await collect(guardedSummary(from(summary), fb))
  check(out.length === 4 && out[1].text === '## Summary\n- ok' && fallbackCalls === 0, 'guarded: a real summary is replayed chunk for chunk, no fallback')
  // a tool call as text falls back
  const disguised = [{ type: 'text-delta', index: 0, text: '<tool_call> <function=bash> <parameter=command> ls' }, { type: 'finish', reason: { kind: 'stop' } }]
  const before = stats.fallbacks
  out = await collect(guardedSummary(from(disguised), fb))
  check(fallbackCalls === 1 && out.some(c => c.text === 'COLD SUMMARY') && !out.some(c => (c.text || '').includes('<tool_call>')) && stats.fallbacks === before + 1,
        'guarded: a tool call in disguise is replaced by the cold path and counted')
  // a real tool-call chunk falls back too
  out = await collect(guardedSummary(from([{ type: 'tool-call-delta', index: 0, id: 'x', name: 'bash', argumentsDelta: '{}' }, { type: 'finish', reason: { kind: 'stop' } }]), fb))
  check(fallbackCalls === 2 && out.some(c => c.text === 'COLD SUMMARY'), 'guarded: a structured tool-call chunk falls back')
  // an errored reply (e.g. context overflow on the alias) falls back
  out = await collect(guardedSummary(from([{ type: 'finish', reason: { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', message: 'too long' } } }]), fb))
  check(fallbackCalls === 3 && out.some(c => c.text === 'COLD SUMMARY'), 'guarded: an error finish (overflow) falls back -- the fit check by effect')
  // a throwing inner stream falls back
  async function * boom () { yield { type: 'text-delta', index: 0, text: 'x' }; throw new Error('socket closed') }
  out = await collect(guardedSummary(boom(), fb))
  check(fallbackCalls === 4 && out.some(c => c.text === 'COLD SUMMARY'), 'guarded: a stream that throws mid-way falls back')
  // an empty reply falls back
  out = await collect(guardedSummary(from([{ type: 'finish', reason: { kind: 'stop' } }]), fb))
  check(fallbackCalls === 5, 'guarded: an empty reply falls back')
  // an ABORTED reply is passed through, never retried
  out = await collect(guardedSummary(from([{ type: 'text-delta', index: 0, text: 'partial' }, { type: 'finish', reason: { kind: 'aborted' } }]), fb))
  check(fallbackCalls === 5 && out.length === 2, 'guarded: an aborted reply is passed through, not retried')

  // through the listener: model rewritten to the alias, tools KEPT, no trim, fallback rebuilds the cold copy
  const L4 = {}
  apply({ on: (n, fn) => { L4[n] = fn } }, { quiet: true, keepToolsVia: KV, toolResultMaxChars: 2000 })
  const big = { role: 'tool', content: 'x'.repeat(5000) }
  const call4 = { purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl', tools: [...tools], maxTokens: 8192, messages: [{ role: 'system', content: 'sys' }, big] }
  let coldSeen = null
  const runtime = { stream: (o) => { coldSeen = o; return from([{ type: 'text-delta', index: 0, text: 'COLD' }, { type: 'finish', reason: { kind: 'stop' } }]) } }
  let innerSeen = false
  const res = L4['llm/stream'].call(runtime, call4, () => { innerSeen = true; return from(disguised) })
  out = await collect(res)
  check(innerSeen && call4.model === 'qwen3.8-27b-vl-compact-notools' && call4.tools.length === 2 && call4.messages[1].content.length === 5000,
        'listener keep: model rewritten to the alias in place, tools kept, tool results NOT trimmed')
  check(coldSeen && coldSeen.model === 'qwen3.8-27b-vl' && !('tools' in coldSeen) && coldSeen.messages[1].content.length < 2200 && out.some(c => c.text === 'COLD'),
        'listener fallback: the cold copy carries the ORIGINAL model, no tools, trimmed results')
  // frozen options on the keep path re-dispatch a copy to the alias
  const f4 = Object.freeze({ purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl', tools: [...tools], maxTokens: 8192, messages: [] })
  let dispatched = []
  const rt2 = { stream: (o) => { dispatched.push(o); return from(summary) } }
  out = await collect(L4['llm/stream'].call(rt2, f4, () => { throw new Error('must not call next() on a frozen keep') }))
  check(dispatched.length === 1 && dispatched[0].model === 'qwen3.8-27b-vl-compact-notools' && dispatched[0].tools.length === 2 && out.length === 4,
        'listener keep (frozen): a copy goes to the alias with its tools, and a good summary is replayed')
  // no runtime handle: the keep path is declined and the certain cold path runs
  const call5 = { purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl', tools: [...tools], maxTokens: 8192, messages: [] }
  let n5 = 0
  L4['llm/stream'].call({}, call5, () => { n5++; return 'inner' })
  check(n5 === 1 && !('tools' in call5) && call5.model === 'qwen3.8-27b-vl', 'listener keep without a runtime handle: tools dropped, model unchanged (the certain path)')
}
// NO TOOL HISTORY ON A TOOLLESS CALL (2026-09-26): pi-ai re-adds `tools: []` when the
// history holds tool calls, and vLLM rejects it -- so the cold-path summary call must
// carry its tool calls and results as text, losing none of their content.
{
  const { flattenToolHistory } = await import('../index.js')
  const hist = [
    { role: 'user', content: [{ type: 'text', text: 'task' }] },
    { role: 'assistant', content: [{ type: 'reasoning', text: 'think' }, { type: 'tool-call', id: 'c1', name: 'bash', arguments: '{"command":"ls"}' }] },
    { role: 'user', content: [{ type: 'tool-result', toolCallId: 'c1', content: [{ type: 'text', text: 'a.py b.py' }], isError: false }] },
    { role: 'user', content: [{ type: 'tool-result', toolCallId: 'c2', content: [{ type: 'text', text: 'boom' }], isError: true }] },
    { role: 'tool', content: 'raw tool text' },
  ]
  const f = flattenToolHistory(hist)
  const types = f.flatMap(m => (Array.isArray(m.content) ? m.content.map(b => b.type) : []))
  check(!types.includes('tool-call') && !types.includes('tool-result') && !f.some(m => m.role === 'tool'),
        'flatten: no tool-call / tool-result block and no tool role survive')
  const all = JSON.stringify(f)
  check(all.includes('bash') && all.includes('ls') && all.includes('a.py b.py') && all.includes('raw tool text'),
        'flatten: tool names, arguments and outputs are all kept as text')
  check(all.includes('(error)') && all.includes('boom'), 'flatten: an error result is marked as an error, with its text')
  check(f[0] === hist[0] && f[1].content[0].type === 'reasoning', 'flatten: messages without tool blocks are untouched; other blocks keep their place')
  check(hist[1].content[1].type === 'tool-call', 'flatten: the input messages are not mutated')
  // the listener applies it to a stripped compaction call, and never to a main-loop call
  const c6 = { purpose: 'compaction', provider: 'spark', model: 'qwen3.8-flash-next', tools: [...tools], maxTokens: 8192, messages: hist }
  run(c6)
  check(!('tools' in c6) && !JSON.stringify(c6.messages).includes('"tool-call"'), 'listener: a stripped compaction call carries no tool history')
  const c7 = { tools: [...tools], messages: hist }
  run(c7)
  check(c7.tools.length === 2 && c7.messages === hist, 'listener: a main-loop call keeps its tools and its history unchanged')
}

// FIFTH JOB (2026-09-28): a title call on a thinking lane goes to its thinking-off alias
{
  const { titleTarget } = await import('../index.js')
  const TR = { 'spark/qwen3.8-flash-next': 'qwen3.8-flash-next-compact' }
  const t1 = { purpose: 'session-title', provider: 'spark', model: 'qwen3.8-flash-next', maxTokens: 64 }
  check(titleTarget(t1, TR) === 'qwen3.8-flash-next-compact', 'title: a mapped lane gets its alias')
  check(titleTarget({ ...t1, model: 'qwen3.6-35b-a3b' }, TR) === undefined, 'title: an unmapped lane of the same box is left alone (no eviction)')
  check(titleTarget({ ...t1, purpose: 'compaction' }, TR) === undefined, 'title: the map never touches a compaction call')
  check(titleTarget({ ...t1, purpose: undefined }, TR) === undefined, 'title: the map never touches a main-loop call')
  const L5 = {}
  apply({ on: (n, fn) => { L5[n] = fn } }, { quiet: true, titleReroute: TR, reroute: { 'spark/qwen3.8-flash-next': 'qwen3.8-flash-next-compact' } })
  // frozen, as dsh-session-title-llm sends it: a copy is re-dispatched to the alias
  let sent = null
  const tf = Object.freeze({ ...t1, messages: [1] })
  const out = L5['llm/stream'].call({ stream: (o) => { sent = o; return 'alias' } }, tf, () => { throw new Error('must re-dispatch a frozen title call') })
  check(out === 'alias' && sent.model === 'qwen3.8-flash-next-compact' && sent.purpose === 'session-title' && sent.maxTokens === 64,
        'title (frozen): a copy goes to the alias with purpose and budget intact')
  let n2 = 0
  L5['llm/stream'].call({ stream: () => { throw new Error('must not loop') } }, sent, () => { n2++ })
  check(n2 === 1, 'title: the alias copy passes straight through on the second dispatch')
  // mutable: the model is changed in place, no re-dispatch
  const tm = { ...t1 }
  let n3 = 0
  L5['llm/stream'].call({}, tm, () => { n3++ })
  check(n3 === 1 && tm.model === 'qwen3.8-flash-next-compact', 'title (mutable): model rewritten in place, next() called')
  // an unmapped title call is untouched
  const tu = { ...t1, provider: 'alder', model: 'qwen3.8-27b-vl' }
  let n4 = 0
  L5['llm/stream'].call({}, tu, () => { n4++ })
  check(n4 === 1 && tu.model === 'qwen3.8-27b-vl', 'title: an alder lane is not rerouted (no titleReroute key)')
  // and a Flash-Next COMPACTION call still takes its own path, not the title one
  const cc = { purpose: 'compaction', provider: 'spark', model: 'qwen3.8-flash-next', tools: [...tools], maxTokens: 8192, messages: [] }
  L5['llm/stream'].call({}, cc, () => {})
  check(cc.model === 'qwen3.8-flash-next-compact' && !('tools' in cc), 'title map does not change the compaction reroute')
}

// SIXTH JOB (2026-09-29): a lane's summary cap is raised; nothing else is touched
{
  const SM = { 'spark/qwen3.8-flash-next': 16384 }
  const RRF = { 'spark/qwen3.8-flash-next': 'qwen3.8-flash-next-compact' }
  const base = { purpose: 'compaction', provider: 'spark', model: 'qwen3.8-flash-next', tools: [...tools], maxTokens: 8192, messages: [] }
  const c1 = classify({ ...base }, RRF, 0, {}, SM)
  check(c1.maxTokens === 16384 && c1.model === 'qwen3.8-flash-next-compact', 'cap: a mapped lane gets 16384 alongside its reroute')
  check(classify({ ...base, model: 'qwen3.6-35b-a3b' }, RRF, 0, {}, SM).maxTokens === undefined, 'cap: an unmapped lane keeps its cap')
  check(classify({ ...base, maxTokens: 20000 }, RRF, 0, {}, SM).maxTokens === undefined, 'cap: never LOWERS a larger request')
  check(classify({ ...base, purpose: undefined }, RRF, 0, {}, SM).action === 'pass', 'cap: a main-loop call is untouched')
  check(classify({ ...base, tools: undefined }, {}, 0, {}, SM).action !== 'pass', 'cap alone is enough to act (no tools, no reroute)')
  const L6 = {}
  const errs = []; const ce = console.error; console.error = (m) => errs.push(String(m))
  apply({ on: (n, fn) => { L6[n] = fn } }, { quiet: true, reroute: RRF, summaryMaxTokens: { ...SM, 'alder/x': '16384' } })
  console.error = ce
  check(errs.some((e) => e.includes('summaryMaxTokens.alder/x') && e.includes('ignored')), 'cap: a quoted number is named and ignored, not silently used')
  // mutable (what dsh-compaction-basic sends): raised in place
  const m1 = { ...base }
  L6['llm/stream'].call({}, m1, () => {})
  check(m1.maxTokens === 16384 && m1.model === 'qwen3.8-flash-next-compact' && !('tools' in m1), 'listener (mutable): cap raised in place, rerouted, tools dropped')
  // frozen: the re-dispatched copy carries it
  let sent = null
  L6['llm/stream'].call({ stream: (o) => { sent = o } }, Object.freeze({ ...base }), () => { throw new Error('frozen must re-dispatch') })
  check(sent && sent.maxTokens === 16384, 'listener (frozen): the copy carries the raised cap')
  // keep path (alias) and its cold fallback both carry it
  const L7 = {}
  apply({ on: (n, fn) => { L7[n] = fn } }, { quiet: true, keepToolsVia: { 'alder/qwen3.8-27b-vl': 'vl-notools' }, summaryMaxTokens: { 'alder/qwen3.8-27b-vl': 12000 } })
  const k1 = { purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl', tools: [...tools], maxTokens: 8192, messages: [] }
  L7['llm/stream'].call({ stream: () => (async function * () {})() }, k1, () => (async function * () {})())
  check(k1.model === 'vl-notools' && k1.maxTokens === 12000, 'listener keep: the alias call carries the raised cap')
  // and a lane with no entry is byte-for-byte what it was
  const u1 = { purpose: 'compaction', provider: 'alder', model: 'qwen3.8-27b-vl', maxTokens: 8192, messages: [] }
  L6['llm/stream'].call({}, u1, () => {})
  check(u1.maxTokens === 8192, 'cap: an unmapped lane is left at 8192')
}

// THE TRIM REALLY TRIMS on the paths Flash-Next takes (2026-09-29). Trim ran AFTER
// flatten on the mutate and redispatch paths, and flatten turns results into text
// blocks the trim does not recognise: nothing was cut since 09-26 while the log
// said "capping N tool result(s)". Driven through the listener, both log shapes.
{
  const RRF = { 'spark/qwen3.8-flash-next': 'qwen3.8-flash-next-compact' }
  const callMsg = { role: 'assistant', content: [{ type: 'tool-call', id: 'c1', name: 'bash', arguments: { command: 'ls' } }] }
  const v3big = { role: 'user', content: [{ type: 'tool-result', toolCallId: 'c1', content: [{ type: 'text', text: 'y'.repeat(5000) }] }] }
  const v4big = { role: 'tool', toolCallId: 'c1', content: [{ type: 'text', text: 'z'.repeat(5000) }] }
  const longest = (msgs) => Math.max(0, ...JSON.stringify(msgs).split(/[^yz]/).map(r => r.length))
  const marks = (msgs) => (JSON.stringify(msgs).match(/omitted before summarization/g) || []).length
  const logs = []; const ce = console.error; console.error = (m) => logs.push(String(m))
  const L8 = {}
  apply({ on: (n, fn) => { L8[n] = fn } }, { reroute: RRF, toolResultMaxChars: 2000 })
  console.error = ce
  for (const [shape, big] of [['v3', v3big], ['v4', v4big]]) {
    const m = { purpose: 'compaction', provider: 'spark', model: 'qwen3.8-flash-next', tools: [...tools], maxTokens: 8192, messages: [callMsg, big] }
    logs.length = 0; console.error = (x) => logs.push(String(x))
    L8['llm/stream'].call({}, m, () => {})
    console.error = ce
    check(!('tools' in m) && longest(m.messages) <= 2000 && marks(m.messages) === 1,
          `trim (mutable, ${shape}): the 5000-char result goes out capped at 2000 with one marker (longest run ${longest(m.messages)})`)
    check(logs.some(l => l.includes('capped 1 tool result(s) at 2000 chars')), `trim (mutable, ${shape}): the log reports the cut actually made`)
    let sent = null
    L8['llm/stream'].call({ stream: (o) => { sent = o } }, Object.freeze({ purpose: 'compaction', provider: 'spark', model: 'qwen3.8-flash-next', tools: [...tools], maxTokens: 8192, messages: [callMsg, big] }), () => { throw new Error('frozen must re-dispatch') })
    check(sent && !('tools' in sent) && longest(sent.messages) <= 2000 && marks(sent.messages) === 1,
          `trim (frozen, ${shape}): the re-dispatched copy is capped too`)
  }
  // nothing oversized: no marker, and the log does not claim a cut
  const small = { purpose: 'compaction', provider: 'spark', model: 'qwen3.8-flash-next', tools: [...tools], maxTokens: 8192,
                  messages: [callMsg, { role: 'tool', toolCallId: 'c1', content: [{ type: 'text', text: 'ok' }] }] }
  logs.length = 0; console.error = (x) => logs.push(String(x))
  L8['llm/stream'].call({}, small, () => {})
  console.error = ce
  check(marks(small.messages) === 0 && !logs.some(l => l.includes('capped')), 'trim: a small result is untouched and no cut is claimed')
}
console.log(failed ? `${failed} FAILED` : 'ALL PASSED'); process.exit(failed ? 1 : 0)
