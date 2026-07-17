"""Build and raw-exec smoke test the NumPy SL-0 Kaggle package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run([sys.executable, "scripts/build_sl0_submission.py"], cwd=ROOT, check=True)
    package = ROOT / "final_submissions/sl0_shared_stage1"
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

    finally:
        os.chdir(old)
    with zipfile.ZipFile(ROOT / "final_submissions/sl0_shared_stage1.zip") as archive:
        names = set(archive.namelist())
        assert {"main.py", "deck.csv", "model_runtime.py", "sl0_shared_best.npz", "MODEL_INFO.json"} <= names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)

    # Keep Torch and NumPy in separate processes to avoid duplicate OpenMP runtimes.
    torch_code = r'''import json,pathlib,torch
from src.train.features import load_card_tags,sample_features
from src.train.shared_model import SharedModelConfig,SharedPolicyValueNet
from agent.parser import parse_observation
fixtures=json.loads(pathlib.Path("tests/fixtures/observations.json").read_text(encoding="utf-8"))
parsed=next(parse_observation(x) for x in fixtures if x["select"]["minCount"]==x["select"]["maxCount"]==1 and len(x["select"]["option"])>1)
g,o,_,_=sample_features(parsed,load_card_tags())
original=pathlib.PosixPath; pathlib.PosixPath=pathlib.WindowsPath
checkpoint=torch.load("artifacts/sl0_shared_full/best.pt",map_location="cpu",weights_only=False); pathlib.PosixPath=original
model=SharedPolicyValueNet(SharedModelConfig(**checkpoint["model_config"])); model.load_state_dict(checkpoint["model_state"]); model.eval()
deck=[int(x) for x in pathlib.Path("submission/deck.csv").read_text().split()]
batch={"global_features":torch.tensor([g]),"option_features":torch.tensor([o]),"legal_mask":torch.ones((1,len(o)),dtype=torch.bool),"player_deck":torch.tensor([deck]),"player_deck_mask":torch.ones((1,len(deck)),dtype=torch.bool),"opponent_deck":torch.zeros((1,1),dtype=torch.long),"opponent_deck_mask":torch.zeros((1,1),dtype=torch.bool)}
with torch.no_grad(): print(json.dumps(model(batch)["policy_logits"][0].tolist()))'''
    numpy_code = rf'''import json,os,pathlib,sys
package=pathlib.Path(r"{package}"); os.chdir(package); sys.path.insert(0,str(package))
from agent.parser import parse_observation
from model_runtime import SL0Policy
fixtures=json.loads(pathlib.Path(r"{ROOT / 'tests/fixtures/observations.json'}").read_text(encoding="utf-8"))
parsed=next(parse_observation(x) for x in fixtures if x["select"]["minCount"]==x["select"]["maxCount"]==1 and len(x["select"]["option"])>1)
deck=[int(x) for x in pathlib.Path("deck.csv").read_text().split()]
print(json.dumps(SL0Policy(deck).logits(parsed).tolist()))'''
    expected = json.loads(subprocess.run([sys.executable, "-c", torch_code], cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    actual = json.loads(subprocess.run([sys.executable, "-c", numpy_code], cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    assert len(expected) == len(actual)
    max_error = max(abs(left - right) for left, right in zip(expected, actual))
    assert max_error <= 2e-3, max_error
    assert max(range(len(expected)), key=expected.__getitem__) == max(range(len(actual)), key=actual.__getitem__)
    print("OK: NumPy SL-0 raw-exec package and legal-action smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
