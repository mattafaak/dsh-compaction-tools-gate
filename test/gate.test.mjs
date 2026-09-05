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
console.log(failed ? `${failed} FAILED` : 'ALL PASSED'); process.exit(failed ? 1 : 0)
