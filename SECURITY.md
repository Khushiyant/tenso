# Security Policy

## Reporting

Report vulnerabilities privately via GitHub Security Advisories:

  https://github.com/Khushiyant/tenso/security/advisories/new

Please do not file public issues for suspected security bugs.

## Supported releases

The latest minor release receives security fixes. Older minors do not; upgrade to
the latest before reporting.

## Threat model

Tenso's core job is parsing binary packets that may come from an untrusted peer
over a network, a shared-memory segment, or a file on disk. Decoding hostile
input is explicitly in scope. A decoder that panics, reads out of bounds, or
allocates memory proportional to an attacker-controlled field rather than to the
actual payload is a security bug, not a robustness nit.

Specifically in scope:

- Any input that makes a decode path panic, abort, or read/write out of bounds.
- Allocation driven by a declared size rather than the delivered bytes
  (decompression bombs, oversized shape or dimension fields).
- Integrity checks that can be bypassed while still reporting success.
- Cross-implementation divergence: the Rust core, the Python binding, and the
  C ABI must agree byte-for-byte, so a packet accepted by one and rejected by
  another is a bug.

Out of scope:

- Confidentiality and authenticity of the transport. Tenso does not encrypt or
  sign packets; the XXH3 footer detects corruption, not tampering. Use TLS or an
  authenticated channel if your peer is untrusted.
- Denial of service from a peer that is authorized to send you arbitrarily large
  but well-formed packets. Bound that at your transport layer.
- The `tenso-cuda` driver paths behind the non-default `cuda` feature, which are
  not exercised by CI.

## What runs against this

- `cargo fuzz` targets over the header parser and the dense decode path.
- Property tests asserting that arbitrary bytes only ever produce a clean
  exception, never a panic or a crash.
- Conformance fixtures pinned by SHA-256, asserted identical across the Rust
  core, the Python binding, and the C ABI.

See `.github/workflows/test.yml` for what actually executes on every push.
