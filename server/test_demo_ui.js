/*
 * The breed-identity panel, tested against real server responses.
 *
 *   node test_demo_ui.js
 *
 * There is no browser and no framework here. The panel's logic is pulled out
 * of static/demo.html and run against a minimal DOM shim, because the thing
 * worth testing is not that elements appear - it is WHAT THE PANEL SAYS.
 *
 * One sentence in particular has to be right. breed_verified comes back false
 * on every single record, because the exact-breed head measured 38.1% on a
 * source-held-out split and switches itself off. false there means NOT
 * CHECKED. If the panel renders it as "breed mismatch" it accuses every
 * correctly registered animal in the district of being mis-registered, and it
 * does so in the one screen a judge or a vet officer actually reads.
 *
 * So these tests assert on the words.
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

// ---- minimal DOM ---------------------------------------------------------
class Node {
  constructor(tag) {
    this.tag = tag; this.className = ""; this.children = []; this._text = "";
  }
  appendChild(c) { this.children.push(c); return c; }
  set textContent(v) { this._text = v; }
  get textContent() { return this._text; }
  // everything the panel wrote, flattened - this is what a reader sees
  get text() {
    return [this._text, ...this.children.map(c => c.text)]
      .filter(Boolean).join(" ");
  }
  get classes() {
    return [this.className, ...this.children.flatMap(c => c.classes)]
      .filter(Boolean);
  }
}

function loadPanel() {
  const html = fs.readFileSync(
    path.join(__dirname, "static", "demo.html"), "utf8");
  const body = html.split(/<script>/)[1].split(/<\/script>/)[0];
  const ctx = {
    document: { createElement: (t) => new Node(t) },
    console,
  };
  // el() and the panel are what we need; the rest of the file references
  // browser globals we do not have, so only the functions are extracted.
  const wanted = ["function el(", "function pct(", "function renderIdentity("];
  const src = wanted.map(sig => {
    const start = body.indexOf(sig);
    if (start < 0) throw new Error("not found in demo.html: " + sig);
    // walk braces from the first { after the signature
    let i = body.indexOf("{", start), depth = 0;
    for (; i < body.length; i++) {
      if (body[i] === "{") depth++;
      else if (body[i] === "}" && --depth === 0) break;
    }
    return body.slice(start, i + 1);
  }).join("\n");
  vm.createContext(ctx);
  vm.runInContext(src + "\nthis.renderIdentity = renderIdentity;", ctx);
  return ctx.renderIdentity;
}

// ---- assertions ----------------------------------------------------------
let failures = 0;
function check(ok, name, detail) {
  if (ok) { console.log("PASS  " + name); }
  else { failures++; console.log("FAIL  " + name + (detail ? "\n      " + detail : "")); }
}

const renderIdentity = loadPanel();
const render = (res) => renderIdentity(res).text;

// A real server response, captured from POST /session against the ML pipeline.
const REAL = {
  animal_id: "356279812345", breed_registered: "Gir",
  breed_verified: false, breed_verify_confidence: 0.0,
  breed_verify_status: "unverified",
  predicted_species: "cattle", species_confidence: 1.0, species_consistent: true,
  predicted_group: "red_zebu", group_confidence: 0.996,
  group_consistent: true, group_reliable: true,
  quality_passed: true,
};

// --- the accusation test --------------------------------------------------
{
  const node = renderIdentity(REAL);
  const t = node.text.toLowerCase();

  // Structural, not textual. "risk" is the accusing style in this console -
  // the same class the outbreak and escalation banners use. An unverified
  // breed must never be dressed in it. Checking the CSS class rather than
  // the wording is what stops a future rephrasing from quietly turning a
  // "not checked" into something that LOOKS like a finding.
  check(!node.classes.includes("risk"),
    "unverified is not styled as a finding",
    "classes: " + node.classes.join(", "));

  // Textual, but only against an AFFIRMATIVE claim. The panel does contain
  // the word "mismatch" - in the sentence "this is NOT a mismatch" - so a
  // bare keyword search would fail on correct copy. Strip the denials first.
  const affirmative = t
    .replace(/not a mismatch/g, "")
    .replace(/never checked/g, "");
  check(!/mismatch|does not match|incorrect breed|wrong breed/.test(affirmative),
    "unverified never asserts a breed mismatch",
    "panel said: " + t.slice(0, 200));

  check(/never checked|not.*checked|switched off/.test(t),
    "unverified says the breed was not checked");
  check(t.includes("gir"), "the registered breed is still shown");
}

// --- the group head is the shippable signal, so it must be visible --------
{
  const t = render(REAL);
  check(/red zebu/i.test(t), "predicted group is displayed");
  check(t.includes("100%"), "group confidence is displayed as a percentage");
  check(/consistent with gir/i.test(t), "agreement with the record is stated");
  check(/cattle/i.test(t), "species is displayed");
}

// --- a genuine disagreement MUST look different --------------------------
{
  const t = render(Object.assign({}, REAL, {
    predicted_group: "buffalo", group_consistent: false }));
  check(/does not match/i.test(t),
    "a real group disagreement is stated plainly");
  check(/checking the record|human check/i.test(t),
    "and is framed as worth a check, not an automatic correction");
}

// --- an unreliable answer is a hint, never a finding ----------------------
{
  const t = render(Object.assign({}, REAL, {
    predicted_group: "exotic_dairy", group_reliable: false,
    group_confidence: 0.55 }));
  check(/hint/i.test(t),
    "group_reliable false is labelled a hint",
    "panel said: " + t.slice(0, 240));
  check(!/does not match/i.test(t),
    "an unreliable answer does not also accuse the record");
}

// --- breed_verify_status disagree is the only accusing path --------------
{
  const t = render(Object.assign({}, REAL, { breed_verify_status: "disagree" }));
  check(/does not match the registered breed/i.test(t),
    "an explicit disagree verdict does say so");
  check(/not an automatic correction/i.test(t),
    "and still refuses to correct the record itself");
  check(renderIdentity(Object.assign({}, REAL,
      { breed_verify_status: "disagree" })).classes.includes("risk"),
    "a real disagreement IS styled as a finding - the two cases must look "
    + "different, or the distinction is invisible to the reader");
}
{
  const t = render(Object.assign({}, REAL, { breed_verify_status: "agree" }));
  check(/agrees with the registered breed/i.test(t), "agree is stated plainly");
}

// --- quality is reported, not used to refuse ------------------------------
{
  const t = render(Object.assign({}, REAL, { quality_passed: false }));
  check(/still scored/i.test(t),
    "a flagged image says the session was scored anyway");
  check(/lower confidence/i.test(t),
    "and tells the reader to weight it less");
  check(!/rejected|refused/i.test(t.replace(/refuse[sd]? to correct/gi, "")),
    "flagged quality is not described as a rejection");
}

// --- missing data degrades quietly ---------------------------------------
{
  const bare = renderIdentity({});
  check(bare.text.length > 0, "an empty result still renders something");
  check(/no identity signal/i.test(bare.text),
    "and says so rather than showing blanks");
}
{
  const t = render(Object.assign({}, REAL, {
    species_confidence: null, group_confidence: undefined }));
  check(/unknown/.test(t), "a missing confidence reads 'unknown', not 'NaN%'");
  check(!/NaN/.test(t), "no NaN reaches the screen");
}

console.log(failures === 0
  ? "\nALL PASS - breed identity panel"
  : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
