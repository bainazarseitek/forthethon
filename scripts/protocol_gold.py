"""Protocol rules on gold facts: does the formulation work when perception is perfect?

Facts are only the `ok` rows of docs/annotations.md (violation rows are never read as input).
Ground truth is data/ground_truth.json events, including unlabeled ones.

    python3 scripts/protocol_gold.py          # per-rule results
    python3 scripts/protocol_gold.py check    # rule self-check
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASH = {"cash_received", "hand_count", "machine_count"}
COUNT = {"hand_count", "machine_count"}


def load_facts():
    facts, vid = {}, None
    for line in (ROOT / "docs/annotations.md").read_text().splitlines():
        if m := re.match(r"^## ([XХ]-\d+)", line):
            vid = m.group(1).replace("Х", "X")
            facts[vid] = []
            continue
        c = [x.strip() for x in line.split("|")]
        if vid and len(c) > 4 and c[2] == "ok" and re.match(r"\**\d", c[1]):
            t = float(re.findall(r"[\d.]+", c[1])[0])
            facts[vid] += [(t, s.strip()) for s in c[3].split(",") if s.strip()]
    return {v: sorted(f) for v, f in facts.items()}


def no_id_check(f):
    # cash handled for a client before any id_check in that visit
    present = checked = False
    for _, s in f:
        if s == "client_arrives": present, checked = True, False
        elif s == "client_leaves": present = False
        elif s == "id_check": checked = True
        elif s in CASH and present and not checked: return True
    return False


def previous_cash_on_desk(f):
    # cash accepted, client leaves, next client served, cash never stored
    cash = left = False
    for _, s in f:
        if s == "cash_received": cash = True
        elif s == "cash_to_safe": cash = False
        elif s == "client_leaves" and cash: left = True
        elif s in {"client_arrives", "document_handover"} and left: return True
    return False


def stored_without_count(f):
    # collection bag opened, then stored with no count in between
    opened = counted = False
    for _, s in f:
        if s == "bag_opened": opened, counted = True, False
        elif s in COUNT: counted = True
        elif s in {"cash_to_safe", "safe_locked"} and opened and not counted: return True
    return False


RULES = [no_id_check, previous_cash_on_desk, stored_without_count]


def check():
    F = lambda *s: list(enumerate(s))
    assert no_id_check(F("client_arrives", "cash_received"))
    assert not no_id_check(F("client_arrives", "id_check", "cash_received"))
    assert not no_id_check(F("hand_count"))  # cashier alone (X-8 pattern)
    assert no_id_check(F("client_arrives", "id_check", "client_leaves", "client_arrives", "hand_count"))
    assert previous_cash_on_desk(F("cash_received", "client_leaves", "client_arrives"))
    assert not previous_cash_on_desk(F("cash_received", "cash_to_safe", "client_leaves", "client_arrives"))
    assert stored_without_count(F("bag_opened", "safe_locked"))
    assert not stored_without_count(F("bag_opened", "machine_count", "cash_to_safe"))
    print("ok")


def main():
    facts = load_facts()
    gt = {v["video"].replace("Х", "X"): {e["type_id"] for e in v["events"]}
          for v in json.loads((ROOT / "data/ground_truth.json").read_text())["videos"]}
    gt.pop("X-47", None); gt.pop("X-48", None)  # exact copies of X-4/X-5
    for rule in RULES:
        name = rule.__name__
        hit = {v for v in gt if rule(facts.get(v, []))}
        true = {v for v in gt if name in gt[v]}
        print(f"{name:24} TP {len(hit & true)}  FP {len(hit - true)}  FN {len(true - hit)}"
              f"  | FP {sorted(hit - true)}  FN {sorted(true - hit)}")
        for v in sorted(true - hit):
            if not facts.get(v):
                print(f"{'':24} {v}: no ok facts annotated")
    n = sum(len(e) for e in gt.values())
    print(f"rule types account for {sum(r.__name__ in e for r in RULES for e in gt.values())}/{n} (video, type) violations")


if __name__ == "__main__":
    check() if sys.argv[1:] == ["check"] else main()
