# Examples

The examples are generated locally so the repository stays small.

## Generate Demo Datasets

```bash
python3 scripts/create_demo_data.py
```

This creates:

- `examples/demo_hdf5.h5`
- `examples/demo_jsonl/`
- `examples/demo_rlds_lite/`
- `examples/demo_robodm.robodm.zip`
- `examples/sample_audit_report.json`

All files are deterministic and tiny.

## Try the CLI

```bash
robotds detect examples/demo_hdf5.h5
robotds audit examples/demo_hdf5.h5 --out reports/audit.json --html reports/audit.html
robotds validate examples/demo_hdf5.h5
robotds convert examples/demo_hdf5.h5 outputs/demo_jsonl --dst jsonl --verify --overwrite
robotds visualize examples/demo_hdf5.h5 --out reports/viz
robotds benchmark examples/demo_hdf5.h5 --json reports/benchmark.json
robotds doctor
```

Inspect `reports/audit.html` and `reports/viz/index.html` in a browser after generating them.

## Cleanup

```bash
rm -rf reports outputs examples/demo_hdf5.h5 examples/demo_jsonl examples/demo_rlds_lite examples/demo_robodm.robodm.zip
```

