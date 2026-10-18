import argparse
from itertools import product
from pathlib import Path
import numpy as np
from .common import load, save, table, plot
from .core import certificate, risk, joint_probability, update


def prepare(a, c):
    if a.input:
        cases = load(a.input)
    else:
        rng = np.random.default_rng(c["seed"])
        cases = []
        shape = (c["actions"],) * c["agents"]
        for k in range(c["episodes"]):
            culprit = rng.choice(c["agents"], max(1, c["agents"] // 2), replace=False)
            base = np.zeros(shape)
            reward = np.zeros(shape)
            coordination = np.zeros(shape)
            for joint in product(range(c["actions"]), repeat=c["agents"]):
                unsafe = sum(joint[i] != 0 for i in culprit) / len(culprit)
                base[joint] = 0.01 + 0.3 * unsafe
                reward[joint] = 1 - 0.1 * np.mean(np.asarray(joint) == 0)
                coordination[joint] = np.mean(np.asarray(joint) == joint[0])
            models = np.stack(
                [
                    np.clip(base + rng.normal(0, c["perturbation"] * 0.1, shape), 0, 1)
                    for _ in range(c["models"])
                ]
            )
            heldout = np.clip(
                base + rng.normal(0, c["perturbation"] * 0.1, shape), 0, 1
            )
            cases.append(
                dict(
                    id=str(k),
                    policy=np.full(
                        (c["agents"], c["actions"]), 1 / c["actions"]
                    ).tolist(),
                    risk_models=models.tolist(),
                    heldout_risk=heldout.tolist(),
                    allowed=[list(range(c["actions"])) for _ in range(c["agents"])],
                    innocent=[i for i in range(c["agents"]) if i not in culprit],
                    reward=reward.tolist(),
                    coordination=coordination.tolist(),
                    source="generated_finite_coordination",
                )
            )
    for case in cases:
        pp = np.asarray(case["policy"])
        if np.any(pp < 0) or not np.allclose(pp.sum(1), 1):
            raise ValueError("Policies must be normalized categorical distributions")
        if np.shape(case["risk_models"])[1:] != tuple(pp.shape[1] for _ in pp):
            raise ValueError("Risk tensor does not match joint action space")
    save(Path(a.data) / "incidents.json", cases)


def calibrate(a, c):
    cases = load(Path(a.data) / "incidents.json")
    rows = []
    for steps in [5, 10, 20, 40, 80]:
        rates = []
        for case in cases:
            policy = [np.array(p) for p in case["policy"]]
            changed = update(
                policy,
                list(range(len(policy))),
                None,
                case["risk_models"],
                steps,
                c["lr"],
            )
            rates.append(float(risk(changed, np.asarray(case["risk_models"])).max()))
        rows.append(dict(steps=steps, risk=np.mean(rates)))
    selected = min(rows, key=lambda r: abs(r["risk"] - np.mean(c["safety_band"])))
    save(Path(a.output) / "calibration.json", dict(selected=selected, candidates=rows))


def run(a, c, method):
    rows = []
    traces = []
    for case in load(Path(a.data) / "incidents.json"):
        policy = [np.array(p) for p in case["policy"]]
        rr = np.asarray(case["risk_models"])
        cert, witness = certificate(
            policy, rr, case["allowed"], c["alpha"], point=method == "point"
        )
        abstained = cert is None
        subset = (
            list(range(len(policy)))
            if abstained or method == "joint"
            else cert["subset"]
        )
        if method == "attribution":
            baseline = float(risk(policy, rr).max())
            effects = []
            for i in range(len(policy)):
                best = baseline
                for action in case["allowed"][i]:
                    other = [p.copy() for p in policy]
                    other[i] = np.eye(len(other[i]))[action]
                    best = min(best, float(risk(other, rr).max()))
                effects.append(baseline - best)
            subset = sorted(range(len(policy)), key=lambda i: (-effects[i], i))[
                : max(1, len(subset))
            ]
            witness = None
        repaired = update(policy, subset, witness, rr, c["steps"], c["lr"])
        if method in ("care", "point") and cert is not None:
            repaired = witness
        innocent = case["innocent"]
        kl = (
            np.mean(
                [
                    np.sum(
                        policy[i]
                        * np.log(
                            np.maximum(policy[i], 1e-12)
                            / np.maximum(repaired[i], 1e-12)
                        )
                    )
                    for i in innocent
                ]
            )
            if innocent
            else None
        )
        before = joint_probability(policy)
        after = joint_probability(repaired)
        reward = np.asarray(case["reward"])
        coordination = np.asarray(case["coordination"])
        rows.append(
            dict(
                id=case["id"],
                method=method,
                violation_rate=float(np.sum(after * np.asarray(case["heldout_risk"]))),
                innocent_kl=kl,
                updated_fraction=len(subset) / len(policy),
                reward_retention=float(
                    np.sum(after * reward) / max(np.sum(before * reward), 1e-12)
                ),
                coordination_retention=float(
                    np.sum(after * coordination)
                    / max(np.sum(before * coordination), 1e-12)
                ),
                certificate_coverage=cert is not None,
                abstention=abstained,
            )
        )
        traces.append(
            dict(
                id=case["id"],
                certificate=cert,
                authorized_subset=subset,
                before=policy,
                after=repaired,
            )
        )
    output = Path(a.output) / method
    table(output / "episodes.csv", rows)
    save(output / "certificates.json", traces)
    summary = {
        "method": method,
        **{
            k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows)
            else None
            for k in rows[0]
            if k not in ("id", "method")
        },
    }
    lo, hi = c["safety_band"]
    summary["within_safety_band"] = lo <= summary["violation_rate"] <= hi
    save(output / "metrics.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "calibrate", "run", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/incidents")
    p.add_argument("--output", default="runs")
    p.add_argument(
        "--method", choices=["care", "point", "joint", "attribution"], default="care"
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "calibrate":
        calibrate(a, c)
    elif a.command == "run":
        run(a, c, a.method)
    elif a.command == "compare":
        rows = [run(a, c, m) for m in ["joint", "attribution", "point", "care"]]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
