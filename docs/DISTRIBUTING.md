# Distributing tenso

tenso ships to several audiences from one repository. Everything below is wired
into `.github/workflows/release.yml` and fires on the **same release** that
`semantic-release` cuts when commits land on `main`.

| Audience | Artifact | Channel | Automation |
| --- | --- | --- | --- |
| Python | `tenso` wheel + sdist | **PyPI** | fully automated (`publish`) |
| Rust | `tenso` / `-device` / `-cuda` / `-ffi` / `-bus` | **crates.io** | automated once the token is set (`publish-crates`) |
| C / C++ | per-OS `.so`/`.a`/`.dylib`/`.dll` + `tenso.h`/`tenso.hpp` | **GitHub Release assets** | fully automated (`release-binaries`) |
| ROS 2 | `tenso_ros`, `tenso_msgs` | **rosdistro / `bloom`** | manual (your ROS account) |

Versions are **lockstep**: the crates inherit `[workspace.package].version`, and
the `publish-crates` job rewrites it (and the internal dep versions) to the
release version before publishing, so crates.io and PyPI share one version.

## 1. Python — PyPI (already live)
No action needed. `release` → `build-wheels` → `build-sdist` → `publish`
(`pypa/gh-action-pypi-publish`, OIDC/trusted-publishing via `id-token: write`).

## 2. Rust — crates.io (one-time setup, then automatic)
The `publish-crates` job is **opt-in**: it no-ops until the token exists, so it
never fails a release before you're ready.

1. Create a crates.io API token (crates.io → Account Settings → API Tokens) with
   publish scope.
2. Add it as a repo secret named **`CARGO_REGISTRY_TOKEN`**
   (Settings → Secrets and variables → Actions → New repository secret).
3. The next release publishes all five crates **in dependency order**
   (`tenso` → `tenso-device` → `tenso-cuda` / `-ffi` / `-bus`); `cargo`
   waits for each to index before the dependents resolve it.

First publish only: crates.io requires each crate name to be available — if any
name is taken, rename in `Cargo.toml` before the first release. To verify
locally without publishing: `cargo publish --dry-run -p tenso`.

Consumers then use: `cargo add tenso` (and `tenso-ffi` for the C ABI, etc.).

## 3. C / C++ — prebuilt binaries (automatic)
`release-binaries` builds `tenso-ffi --release` on Linux / macOS / Windows and
attaches `tenso-ffi-<platform>.tar.gz` to the GitHub Release. Each archive holds
`lib/` (shared + static lib) and `include/` (`tenso.h`, `tenso.hpp`).

Consumers download the archive for their platform and link against the lib +
header — no Rust toolchain or cargo required. (Building from source also works:
`cargo build -p tenso-ffi --release`.)

## 4. ROS 2 — rosdistro via bloom (manual)
The `ros/tenso_ros` and `ros/tenso_msgs` packages carry a `package.xml` and are
bloom-ready, but releasing into the ROS index needs your ROS identity, so it is
not automated:

```bash
# one-time per package, from your ROS environment
pip install bloom
bloom-release tenso_ros --rosdistro <distro> --track <distro>
# follow the prompts; bloom opens a PR against ros/rosdistro
```

Until then, users build from source in a colcon workspace:
`colcon build --packages-select tenso_msgs tenso_ros`.

## Cutting a release
Merging to `main` with `feat:`/`fix:` commits triggers `semantic-release`, which
bumps the version, tags, and creates the GitHub Release; the jobs above then
publish to each configured channel. There is no separate "release" button — the
merge is the release.
