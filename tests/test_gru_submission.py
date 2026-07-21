"""Build, numerically verify, and smoke-test the NumPy SL-1 package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def run(code: str, cwd: Path) -> object:
    result = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main() -> int:
    subprocess.run([sys.executable, "scripts/build_gru_submission.py"], cwd=ROOT, check=True)
    package = ROOT / "final_submissions/sl1_gru_seed20260721_stage1"
    with zipfile.ZipFile(package.with_suffix(".zip")) as archive:
        names = set(archive.namelist())
        assert {"main.py", "deck.csv", "gru_model_runtime.py", "sl1_gru_best.npz", "MODEL_INFO.json"} <= names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)

    old = Path.cwd()
    try:
        os.chdir(package)
        env = {"__builtins__": __builtins__}
        exec((package / "main.py").read_text(encoding="utf-8"), env)
        agent = env["agent"]
        assert len(agent(None)) == 60
        fixtures = json.loads((ROOT / "tests/fixtures/observations.json").read_text(encoding="utf-8"))
        for observation in fixtures[:50]:
            action = agent(observation)
            select = observation["select"]
            assert select["minCount"] <= len(action) <= select["maxCount"]
            assert len(action) == len(set(action))
            assert all(0 <= value < len(select["option"]) for value in action)
        policy = env["get_policy"]()
        assert len(policy._temporal_inputs) >= 0
        policy.reset()
        assert policy._temporal_inputs == [] and policy._previous_global is None
    finally:
        os.chdir(old)

    torch_code = r'''import json,math,pathlib,torch
from src.train.features import load_card_tags,sample_features
from agent.parser import parse_observation
from src.train.sequence_model import SequenceModelConfig,SequencePolicyValueNet
original=pathlib.PosixPath; pathlib.PosixPath=pathlib.WindowsPath
checkpoint=torch.load("artifacts/sl1_gru_seed20260721/best.pt",map_location="cpu",weights_only=False); pathlib.PosixPath=original
model=SequencePolicyValueNet(SequenceModelConfig(**checkpoint["model_config"])); model.load_state_dict(checkpoint["model_state"]); model.eval()
raw=json.loads(pathlib.Path("tests/fixtures/observations.json").read_text(encoding="utf-8"))
parsed=[parse_observation(x) for x in raw if x["select"]["minCount"]==x["select"]["maxCount"]==1 and len(x["select"]["option"])>1][:3]
tags=load_card_tags(); deck=[int(x) for x in pathlib.Path("submission/deck.csv").read_text().split()]
globals=[]; options=[]
for p in parsed:
 g,o,_,_=sample_features(p,tags); globals.append(torch.tensor(g,dtype=torch.float32)); options.append(torch.tensor(o,dtype=torch.float32))
embedding=model.card_embedding(torch.tensor(deck)).mean(0); opponent=torch.zeros_like(embedding); deck_context=model.deck_encoder(torch.cat([embedding,opponent]))
temporal=[]; previous_action=torch.zeros(options[0].shape[1]); previous_global=None; previous_turn=None
for i,p in enumerate(parsed):
 state=model.state_encoder(globals[i]); base=model.context(torch.cat([state,deck_context]))
 if previous_global is None: transition=torch.zeros(24)
 else: transition=torch.cat([(globals[i][:22]-previous_global[:22]).clamp(-1,1),torch.tensor([float(p.turn!=previous_turn),float(bool(p.logs))])])
 temporal.append(torch.cat([base,model.transition_encoder(transition),model.previous_action_encoder(previous_action)]))
 previous_global=globals[i]; previous_turn=p.turn; previous_action=options[i][0]
sequence=torch.stack(temporal).unsqueeze(0); output,_=model.gru(sequence); context=base+model.temporal_projection(output[0,-1]); encoded=model.option_encoder(options[-1]); logits=(encoded*context).sum(-1)/math.sqrt(context.shape[-1])+model.option_bias(encoded).squeeze(-1)
print(json.dumps({"logits":logits.tolist(),"count":len(parsed)}))'''
    numpy_code = rf'''import json,os,pathlib,sys,numpy as np
package=pathlib.Path(r"{package}"); os.chdir(package); sys.path.insert(0,str(package))
from agent.parser import parse_observation
from gru_model_runtime import GRUPolicy,sample_features
raw=json.loads(pathlib.Path(r"{ROOT / 'tests/fixtures/observations.json'}").read_text(encoding="utf-8"))
parsed=[parse_observation(x) for x in raw if x["select"]["minCount"]==x["select"]["maxCount"]==1 and len(x["select"]["option"])>1][:3]
deck=[int(x) for x in pathlib.Path("deck.csv").read_text().split()]; policy=GRUPolicy(deck)
last=None
for p in parsed:
 g,o=sample_features(p,policy.tags); prev=policy._previous_action if policy._previous_action is not None else np.zeros(o.shape[1],dtype=np.float32); transition=policy._transition(g,p.turn,bool(p.logs)); logits,item=policy._forward_step(g,o,transition,prev); policy._temporal_inputs.append(item); policy._previous_global=g.copy(); policy._previous_turn=p.turn; policy._previous_action=o[0].copy(); last=logits
print(json.dumps({{"logits":last.tolist(),"count":len(parsed)}}))'''
    expected = run(torch_code, ROOT)
    actual = run(numpy_code, ROOT)
    assert expected["count"] == actual["count"] >= 1
    errors = [abs(left - right) for left, right in zip(expected["logits"], actual["logits"])]
    assert max(errors) <= 3e-3, max(errors)
    assert max(range(len(expected["logits"])), key=expected["logits"].__getitem__) == max(
        range(len(actual["logits"])), key=actual["logits"].__getitem__
    )
    print(f"OK: NumPy SL-1 package, reset/mask smoke, max logit error={max(errors):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
