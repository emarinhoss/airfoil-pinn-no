"""Stage S3: the PINN on the showcase cases (spec 7.3-7.4). Resumes by skipping finished files."""

import json
from pathlib import Path

from apno.data.predictions import write_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN, Section
from apno.models.pinn import PinnCase
from apno.models.train_pinn import PinnConfig, evaluate_on_grid, train

PRED, RES = Path("runs/predictions"), Path("results/s3")


def plan():
    show = json.loads(Path("results/s1/showcase.json").read_text())["cases"]
    sup = [c for c in show if c["regime"] == "supercritical"]
    sub = [c for c in show if c["regime"] == "subcritical"]
    runs = [("euler", c["case_id"], PinnConfig()) for c in show]
    runs += [("visc1e-3", c["case_id"], PinnConfig(viscosity=1e-3)) for c in sup]
    if sup:
        top = max(sup, key=lambda c: c["max_surface_mach"])["case_id"]
        runs.append(("visc3e-4", top, PinnConfig(viscosity=3e-4)))
    if sub:
        runs.append(("nokutta", sub[0]["case_id"], PinnConfig(kutta=False)))
    return runs


def main():
    specs = {s.case_id: s for s in case_specs("test_in", TRAIN)}
    RES.mkdir(parents=True, exist_ok=True)
    record_path = RES / "pinn_runs.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for variant, cid, cfg in plan():
        path = PRED / f"pinn-{variant}-{cid}.h5"
        if path.exists():
            continue
        s = specs[cid]
        pinn, rec = train(PinnCase(s.mach, s.alpha_deg, Section(s.t, s.m)), cfg)
        q = evaluate_on_grid(pinn, TRAIN)
        write_predictions(path, [cid], q[None], {
            "method": "pinn", "variant": variant, "seed": cfg.seed, "n_train": 0,
            "train_wall_s": rec["wall_s"], "train_cpu_s": rec["cpu_s"], "query_s": rec["wall_s"]})
        record[f"{variant}/{cid}"] = {"wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"],
                                      "final": rec["final"], "history": rec["history"]}
        record_path.write_text(json.dumps(record, indent=2))
        print(variant, cid, rec["final"], flush=True)


if __name__ == "__main__":
    main()
