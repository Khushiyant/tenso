# tenso fuzz targets (atheris)

Coverage-guided fuzz harnesses for the Python entry points
(`tenso.loads` and `tenso.get_packet_info`). Fuzzing is opt-in;
`atheris` is **not** declared as a project dependency.

## Setup

```sh
pip install atheris
# atheris ships an LLVM-libFuzzer-linked CPython on most platforms;
# see https://github.com/google/atheris if pip install fails.
```

## Run

```sh
# from the repo root
python pyfuzz/fuzz_loads.py            pyfuzz/seeds/
python pyfuzz/fuzz_get_packet_info.py  pyfuzz/seeds/
```

Cap a single fuzz session at 60 seconds, useful for CI smoke runs:

```sh
python pyfuzz/fuzz_loads.py -max_total_time=60 pyfuzz/seeds/
```

## Corpus

`pyfuzz/seeds/` is checked into git and produced by
`pyfuzz/_make_seeds.py`. Regenerate after protocol changes:

```sh
python pyfuzz/_make_seeds.py
```

## Crash artifacts

By default libFuzzer writes crashing inputs to the current working
directory as `crash-<sha1>`. Reproduce with:

```sh
python pyfuzz/fuzz_loads.py crash-<sha1>
```
