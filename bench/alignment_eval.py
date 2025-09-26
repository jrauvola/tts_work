import json
from typing import List


def mean_abs_error_ms(pred: List[dict], ref: List[dict]) -> float:
    # assumes same ordering by char_index
    ref_map = {r["char_index"]: r for r in ref}
    errors = []
    for p in pred:
        r = ref_map.get(p["char_index"])  # missing allowed
        if not r:
            continue
        errors.append(abs(float(p["start_ms"]) - float(r["start_ms"])))
    return sum(errors) / max(1, len(errors))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="path to predicted alignments json")
    parser.add_argument("--ref", required=True, help="path to reference alignments json")
    args = parser.parse_args()
    pred = json.load(open(args.pred))
    ref = json.load(open(args.ref))
    print({"mae_ms": mean_abs_error_ms(pred, ref)})


