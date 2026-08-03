#!/usr/bin/env node
/* selftest.js — headless check of index.html's render layer.
 *
 * WHY THIS FILE EXISTS. nfl-daily.yml published the board with NO gate of any kind,
 * and the render block -- ~125 lines that turn slate.json into the page -- had no test
 * at all. The sibling repos have already produced two defects in exactly this layer:
 * a panel written to a DOM id that does not exist (MLBTool, for months), and a market
 * comparison that quoted the model over one game set and the market over another.
 *
 * The second one was live here too. The Method strip printed
 *     model 64.9% · market 66.4% · disagreements 679 @ model 45.1% right
 * where 64.9% is over all 4,350 backtested games and 66.4% is over the 4,349 that
 * carried a closing line -- and it never said how often the MARKET was right on those
 * 679 games where the model claimed to know better. It is 54.9%. That is the whole
 * verdict on fading the price with this model, and it was left to the reader to
 * subtract.
 *
 * This harness extracts the SHIPPED <script> block, stubs the DOM with a
 * getElementById that returns null for ids not in the markup (inventing elements on
 * demand is what hides the dead-panel bug class), and renders the real committed slate.
 */
const fs = require('fs');
const path = require('path');
const HERE = __dirname;

const html = fs.readFileSync(path.join(HERE, 'index.html'), 'utf8');
let failures = 0;
const fail = m => { console.log('  FAIL: ' + m); failures++; };
const ok = m => console.log('  ok: ' + m);
const check = (c, m) => (c ? ok(m) : fail(m));

const blocks = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = blocks.filter(b => /function render\s*\(/.test(b));
if (main.length !== 1) { fail(`expected 1 script block defining render(), found ${main.length}`); process.exit(1); }
const src = main[0].replace(/fetch\('data\/slate\.json'[\s\S]*$/, '');

const REAL_IDS = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));

function runRender(slate) {
  const els = {};
  const mk = id => ({ id, innerHTML: '', textContent: '', style: {}, className: '', value: '',
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    appendChild() {}, querySelectorAll: () => [], querySelector: () => null,
    addEventListener() {}, checked: false, dataset: {} });
  const doc = {
    getElementById: id => (REAL_IDS.has(id) ? (els[id] || (els[id] = mk(id))) : null),
    querySelectorAll: () => [], querySelector: () => null,
    createElement: () => mk('_tmp'), addEventListener() {}, body: mk('body'),
  };
  const errors = [];
  const sandbox = {
    document: doc,
    window: { addEventListener() {}, location: { hash: '' } },
    location: { hash: '' },
    console: { log() {}, warn() {}, info() {}, error: (...a) => errors.push(a.map(String).join(' ')) },
    fetch: () => Promise.reject(new Error('no network in selftest')),
    setTimeout, clearTimeout, Math, JSON, Date, Number, String, Array, Object, Set, Map,
    Boolean, isFinite, isNaN, parseFloat, parseInt,
    MutationObserver: function () { this.observe = function () {}; this.disconnect = function () {}; },
  };
  const names = Object.keys(sandbox);
  const fn = new Function(...names, '__SLATE__', '__ELS__', src + '\n;render(__SLATE__); return 0;');
  fn(...names.map(n => sandbox[n]), slate, els);
  return { els, errors, html: id => (els[id] ? els[id].innerHTML : null) };
}

const SLATE = JSON.parse(fs.readFileSync(path.join(HERE, 'data', 'slate.json'), 'utf8'));
const clone = o => JSON.parse(JSON.stringify(o));

/* --------------------------------- 0) no renderer writes to a nonexistent element */
console.log('0) every id the render layer writes to is really in the markup');
const targets = [...src.matchAll(/\$\('([A-Za-z0-9_-]+)'\)/g)].map(m => m[1]);
const missing = [...new Set(targets)].filter(id => !REAL_IDS.has(id));
check(missing.length === 0,
  'no $(id) points at an element that does not exist' +
  (missing.length ? ' — MISSING: ' + missing.join(', ') : ''));

/* --------------------------------------------- 1) the committed slate renders */
console.log('1) render the committed slate.json');
let r = runRender(clone(SLATE));
check(r.errors.length === 0, 'render() threw nothing' + (r.errors.length ? ': ' + r.errors[0] : ''));
check((r.html('btBody') || '').length > 40, 'the Method backtest strip painted');

/* ------------------------------- 2) the backtest strip compares like with like */
console.log('2) the backtest strip compares one game set');
// Force acc and acc_mkt apart so the panel cannot pass by quoting the wrong one.
const S = clone(SLATE);
S.backtest = { n: 4350, acc: 64.9, acc_mkt: 61.1, brier: 0.2202, n_mkt: 4349,
               market_acc: 66.4, n_disagree: 679,
               model_right_in_disagree: 45.1, market_right_in_disagree: 54.9 };
let h = runRender(S).html('btBody') || '';
check(/model <b>61\.1%<\/b>/.test(h),
  'it quotes the model on the MARKET\'s subset (acc_mkt), not over all games');
check(/market <b>66\.4%<\/b>/.test(h), 'and the market beside it');
check(/4349/.test(h), 'it names the game set both figures are over');
check(/market <b>54\.9%<\/b>/.test(h),
  'it states how often the MARKET was right on the disagreements');
check(/fading it with this model loses/.test(h),
  'a model that loses its own disagreements says so in words');
check(!/undefined|NaN/.test(h), 'no "undefined"/"NaN" in the strip');

// A model that WINS its disagreements must not get the warning.
console.log('3) the warning is earned, not automatic');
const S2 = clone(S);
S2.backtest.model_right_in_disagree = 58.0; S2.backtest.market_right_in_disagree = 42.0;
h = runRender(S2).html('btBody') || '';
check(!/fading it with this model loses/.test(h),
  'a model that wins its disagreements is NOT warned about (test is market>model, not model<50)');

// A slate published before acc_mkt existed must still render.
console.log('4) an older slate still renders');
const S3 = clone(SLATE);
delete S3.backtest.acc_mkt; delete S3.backtest.market_right_in_disagree;
h = runRender(S3).html('btBody') || '';
check(!/undefined|NaN/.test(h), 'no "undefined"/"NaN" with the new fields absent');
check(/model <b>64\.9%<\/b>/.test(h), 'it falls back to the overall accuracy rather than blank');

// No backtest block at all is a legitimate state (a slate built before it ran).
console.log('5) missing blocks do not take the page down');
const S4 = clone(SLATE); delete S4.backtest; delete S4.cal;
r = runRender(S4);
check(r.errors.length === 0, 'render() survives a slate with no backtest and no cal');

/* ------------------------- 6) the LIVE market panel compares like with like too */
console.log('6) the live calibration panel compares one game set');
// The committed slate carries cal.n === 0 and no cal.market, so sections 1-5 never
// touch this branch -- it is the one that will actually render every week once games
// start grading, and it carried the same apples-to-oranges bug as the Method strip.
// The fixture forces model_acc (58.3) apart from the overall acc (61.5) so the panel
// cannot pass by quoting c.acc, and puts the market ahead on the disagreements.
{
  const S5 = clone(SLATE);
  S5.cal = { n: 26, weeks: 3, acc: 61.5, brier: 0.2311, ties: 1,
             buckets: [{ bucket: '60-70', n: 12, pred: 64.0, actual: 58.3 }],
             market: { n: 24, acc: 66.7, model_acc: 58.3, disagree_n: 9,
                       disagree_model_right: 33.3, disagree_market_right: 66.7 } };
  const g = runRender(S5);
  check(g.errors.length === 0, 'render() threw nothing on a graded slate'
    + (g.errors.length ? ': ' + g.errors[0] : ''));
  const hh = g.html('calBody') || '';
  check(hh.length > 40, 'the live calibration panel painted');
  check(/model <b>58\.3%<\/b>/.test(hh),
    "it quotes the model on the market's own subset (model_acc), not c.acc over every graded game");
  check(!/model <b>61\.5%<\/b>/.test(hh), 'and specifically not the all-games figure');
  check(/market <b>66\.7%<\/b>/.test(hh), 'the market beside it, over that same subset');
  check(/on the <b>24<\/b> graded games/.test(hh), 'it names the game set both figures are over');
  check(/market right <b>66\.7%<\/b>/.test(hh),
    'it states how often the MARKET was right on the disagreements');
  check(/model right <b[^>]*>33\.3%<\/b>/.test(hh), 'and the model\'s own rate on them');
  check(!/undefined|NaN/.test(hh), 'no "undefined"/"NaN" in the live panel');

  // A ledger written before the new fields existed must degrade, not print undefined.
  const S6 = clone(S5);
  delete S6.cal.market.model_acc; delete S6.cal.market.disagree_market_right;
  const h6 = runRender(S6).html('calBody') || '';
  check(!/undefined|NaN/.test(h6), 'an older market block renders with no "undefined"/"NaN"');
  check(/model <b>61\.5%<\/b>/.test(h6), 'and falls back to the overall accuracy rather than blank');

  // cal.n === 0 (the committed state) must leave the placeholder alone, not blank it.
  const S7 = clone(SLATE); S7.cal = { n: 0, weeks: 0 };
  check(runRender(S7).errors.length === 0, 'a slate with zero graded games does not throw');
}

/* ---------------------------------------------------- 7) no stale-cache fetches */
console.log('7) the slate cannot be served from browser cache');
const fetches = [...html.matchAll(/fetch\(\s*'(data\/[^']+)'([^)]*)\)/g)];
check(fetches.length > 0, 'found the data fetches');
const cacheable = fetches.filter(m => !/no-store/.test(m[2])).map(m => m[1]);
check(cacheable.length === 0,
  "every data fetch passes cache:'no-store'" +
  (cacheable.length ? ' — CACHEABLE: ' + cacheable.join(', ') : ''));

console.log(failures ? `\nNFL UI SELFTEST: ${failures} FAILURE(S)`
                     : '\nNFL UI SELFTEST PASS — the Method strip states the market\'s accuracy on '
                       + 'the disagreements over the same games, instead of leaving it to be derived');
process.exit(failures ? 1 : 0);
