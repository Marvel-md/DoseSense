/*
 * Render harness.
 *
 * The dashboard has no build step, so there is no compiler to catch a typo in a
 * template literal or a field that the API does not actually return. A syntax
 * check proves the file parses; it does not prove that renderDetail survives a
 * patient with no symptom questionnaires, an abstained assessment or an empty
 * timeline.
 *
 * This loads the real script into a minimal fake DOM, feeds it payloads fetched
 * from the running API, and calls the render functions directly. Any undefined
 * property access or thrown error fails the run. It is not a substitute for
 * looking at the page, but it does mean the page cannot be silently broken by a
 * change to an API field name.
 */

const fs = require("fs");
const path = require("path");
const liveFetch = global.fetch;
if (!liveFetch) throw new Error("Node.js 18 or newer is required.");
const baseURL = (process.env.DOSESENSE_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
async function getJSON(route) {
  const response = await liveFetch(baseURL + route, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`${route}: HTTP ${response.status} — ${await response.text()}`);
  return response.json();
}


const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
const src = html.match(/<script>\n([\s\S]*?)\n<\/script>/)[1];

// ---- minimal DOM ---------------------------------------------------------

function makeEl(id) {
  const el = {
    id,
    _html: "",
    textContent: "",
    value: "",
    disabled: false,
    dataset: {},
    style: {},
    set innerHTML(v) { this._html = String(v); },
    get innerHTML() { return this._html; },
    addEventListener() {},
    onclick: null,
    onkeydown: null,
    setAttribute() {},
  };
  return el;
}

const els = {};
const IDS = ["live", "banner", "main", "q", "state-filters", "tier-filters",
             "barrier-filters", "back", "logged"];
IDS.forEach(i => { els[i] = makeEl(i); });

let queried = [];
global.document = {
  getElementById: id => els[id] || (els[id] = makeEl(id)),
  querySelectorAll: sel => { queried.push(sel); return []; },
};
global.window = { scrollTo() {} };
global.fetch = async () => ({ ok: false, status: 503, json: async () => ({}) });

// ---- load the real script ------------------------------------------------
// The boot IIFE fires on load and fails against the stub fetch, which is fine:
// the failure path is itself something worth exercising.

const load = new Function(`${src}\nreturn {renderList, renderDetail, renderFilters, renderNav, renderPortal, sparkline, lineChart, barChart, S};`);
const app = load();

// ---- exercise ------------------------------------------------------------

const problems = [];
function check(name, fn, target = "main") {
  try {
    fn();
    const out = els[target].innerHTML;
    if (!out || out.length < 200) throw new Error(`rendered only ${out.length} chars`);
    if (/undefined|NaN|\[object Object\]/.test(out)) {
      const m = out.match(/.{0,60}(undefined|NaN|\[object Object\]).{0,60}/);
      throw new Error(`leaked placeholder into output: …${m[0]}…`);
    }
    console.log(`  ok    ${name}  (${out.length.toLocaleString()} chars)`);
  } catch (e) {
    problems.push(`${name}: ${e.message}`);
    console.log(`  FAIL  ${name}  ${e.message}`);
  }
}

async function main() {
// Let the deliberately failed boot request finish before manual render checks.
await new Promise(resolve => setImmediate(resolve));
const health = await getJSON("/api/health");
if (health.status !== "ok") throw new Error(`API is not ready: ${JSON.stringify(health)}`);
const [overview, queuePayload, demo] = await Promise.all([
  getJSON("/api/overview"), getJSON("/api/patients?limit=2000"), getJSON("/api/demo")
]);
const queue = queuePayload.patients;
if (!queue.length) throw new Error("No patients available for render checks.");
if (queue.length !== queuePayload.count) throw new Error("Queue truncated; run this check on a demo cohort of at most 2000 patients.");
const picks = {};
for (const row of queue) picks[row.state] ??= row.patient_id;
for (const c of demo.cases) picks[`demo: ${c.slot}`] = c.patient_id;
const details = {};
for (const [label, pid] of Object.entries(picks)) {
  details[label] = await getJSON(`/api/patients/${encodeURIComponent(pid)}`);
}
console.log(`API ready: ${queue.length} patients; ${demo.cases.length} walkthrough cases.`);
app.S.overview = overview;
app.S.queue = queue;

console.log("\nrender harness");
// renderFilters now returns markup rather than writing to its own element,
// so it is checked on its return value.
try {
  const out = app.renderFilters();
  if (!out || out.length < 80) throw new Error(`returned only ${out ? out.length : 0} chars`);
  if (/undefined|NaN/.test(out)) throw new Error("leaked a placeholder");
  console.log(`  ok    filter row  (${out.length} chars)`);
} catch (e) {
  problems.push(`filter row: ${e.message}`);
  console.log(`  FAIL  filter row  ${e.message}`);
}
check("review queue", () => app.renderList());

check("queue, filtered to nothing", () => {
  app.S.filters = { state: "NOT_A_STATE", tier: null, barrier: null, q: "" };
  app.renderList();
});
app.S.filters = { state: null, tier: null, barrier: null, q: "" };

for (const [state, pid] of Object.entries(picks)) {
  const d = details[state];
  check(`patient detail — ${state} (${pid})`, () => app.renderDetail(d));
}

// Degenerate payloads: a patient with nothing plottable must not crash.
const bare = details[Object.keys(picks)[0]];
check("patient with no series data", () => {
  const d = JSON.parse(JSON.stringify(bare));
  d.series = { refills: [], symptoms: [], labs: [], lab_marker: "HbA1c", lab_unit: "%",
               symptom_scale: "burden" };
  d.timeline = [];
  d.feedback = [];
  d.assessment.observed.facts = [];
  d.assessment.inferred.attribution = null;
  d.assessment.inferred.change_detection = {};
  app.renderDetail(d);
});

// ---- patient portal -----------------------------------------------------
// The portal is a different audience with a different safety requirement, so it
// is checked separately: it must render, and it must never contain the estimate.

{
  const portal = await getJSON(`/api/patients/${encodeURIComponent(queue[0].patient_id)}/portal`);
  app.S.role = "patient";
  app.S.me = portal.patient.patient_id;
  app.S.portal = portal;
  app.S.draft = { doses: null, symptom: null, barriers: [], note: "", date: "2026-09-12" };

  check("role selector", () => app.renderNav(), "nav");
  check("patient portal", () => app.renderPortal());

  check("portal with answers part-filled", () => {
    app.S.draft = { doses: "some", symptom: 7,
                    barriers: (portal.barrier_options || []).slice(0, 2).map(b => b.id),
                    note: "ran out early", date: "2026-09-12" };
    app.renderPortal();
  });

  // Safety: no risk language may reach the patient's screen.
  try {
    const html = els.main.innerHTML;
    const banned = ["adherence_concern", "priority_tier", "SUSTAINED_CONCERN", "CRITICAL",
                    "non-compliant", "did not take", "probability of"];
    const hit = banned.find(b => html.toLowerCase().includes(b.toLowerCase()));
    if (hit) throw new Error(`portal rendered risk language: ${hit}`);
    console.log("  ok    portal shows no risk language");
  } catch (e) {
    problems.push(`portal language: ${e.message}`);
    console.log(`  FAIL  portal language  ${e.message}`);
  }

  check("portal for a patient with no history", () => {
    const bare = JSON.parse(JSON.stringify(portal));
    bare.medications = []; bare.self_reports = []; bare.next_refill_due = null;
    bare.last_collected = null; bare.recent_symptoms = [];
    app.S.portal = bare;
    app.renderPortal();
  });
  app.S.role = "doctor";
}

console.log("");
if (problems.length) {
  console.log(`${problems.length} render failure(s)`);
  process.exit(1);
}
console.log("all render checks passed");


}
main().catch(error => {
  console.error(`Render check failed: ${error.message}`);
  console.error("Start DoseSense first, or set DOSESENSE_BASE_URL to its running API.");
  process.exitCode = 1;
});
