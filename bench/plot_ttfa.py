import csv
import sys


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    print("Concurrency,Model,p50_ms,p95_ms,qps")
    for r in rows:
        print(f"{r['concurrency']},{r['model']},{r['p50_ms']},{r['p95_ms']},{r['qps']}")


if __name__ == "__main__":
    main(sys.argv[1])


