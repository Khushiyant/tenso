# CHANGELOG


## v0.4.7 (2025-12-14)

### Bug Fixes

- Add write_stream to __init__.py and optimize _read_exact function in core.py
  ([`d6d0dd7`](https://github.com/Khushiyant/tenso/commit/d6d0dd74cb876a345e02f54ad47f3bd4ca1940fc))

- Enhance benchmark.py with serialization and I/O tests, add stream read/write functionality
  ([`310c3ff`](https://github.com/Khushiyant/tenso/commit/310c3ff69ea4ffe8dac2233d16506092774d83ab))

- Retry release 0.4.7
  ([`ccdb3a0`](https://github.com/Khushiyant/tenso/commit/ccdb3a09b095747dcb034523882224c595f0c5eb))


## v0.4.6 (2025-12-14)

### Bug Fixes

- Update release workflow name and enhance steps for Python package publishing
  ([`a162131`](https://github.com/Khushiyant/tenso/commit/a16213162fda1ea0048c54e8d9b414064d084284))


## v0.4.5 (2025-12-14)

### Bug Fixes

- Optimize _read_exact function and clean up error handling
  ([`d1b3033`](https://github.com/Khushiyant/tenso/commit/d1b30331e52adb60ed6511ff3e2fcd4e01d7f8b1))


## v0.1.0 (2025-12-14)


## v0.4.4 (2025-12-14)

### Bug Fixes

- Improve response handling in client and server for large packets and add padding support
  ([`bab51d8`](https://github.com/Khushiyant/tenso/commit/bab51d821f2e0297df98d7f16e649a0685fa606a))

- Remove IS_LITTLE_ENDIAN variable from config
  ([`9c77814`](https://github.com/Khushiyant/tenso/commit/9c77814031891db68ba319bbd1234c5f2453c7c6))

- Update package name from tenso-core to tenso in installation instructions
  ([`ebdf752`](https://github.com/Khushiyant/tenso/commit/ebdf7522c40e5d99890e7514be1352e63f943d04))

- Update project description in pyproject.toml
  ([`7a810b2`](https://github.com/Khushiyant/tenso/commit/7a810b2982ee96cdfc1400d85cbccf2472035c12))

- Update version to 0.2.2 and improve alignment check in utility functions
  ([`cb1ce40`](https://github.com/Khushiyant/tenso/commit/cb1ce40b37e2f0968a3223a2c95a213a8e9fbc21))

### Chores

- Add github pages--tenso
  ([`3cf4fd7`](https://github.com/Khushiyant/tenso/commit/3cf4fd7348d270791a4c13f6ed72ef7c37661e97))

- Add image banner to README
  ([`0ed2c27`](https://github.com/Khushiyant/tenso/commit/0ed2c27acbc132960263fd1526659b5c5a90bbc0))

- Add initial implementation of Tenso library with serialization and deserialization for NumPy
  arrays
  ([`3cbe1fd`](https://github.com/Khushiyant/tenso/commit/3cbe1fd868d71a613629b2ae376115deaf69fd22))

- Implement core serialization protocol in `core.py` - Add example client and server for tensor
  communication - Create README with usage instructions and features - Include tests for round-trip
  serialization and file I/O - Add .gitignore and LICENSE files - Update pyproject.toml with
  dependencies and build system

### Continuous Integration

- Add GitHub Actions workflows for semantic release and testing
  ([`8818f7a`](https://github.com/Khushiyant/tenso/commit/8818f7a9092a3858f760106499d899f91acab47f))

- Ensure pytest is installed in the CI workflow
  ([`9adfbc4`](https://github.com/Khushiyant/tenso/commit/9adfbc4bbcb524c0bfd297df94eac36016e8aa64))

- Remove incorrect root_options from semantic release workflow
  ([`f66bd6f`](https://github.com/Khushiyant/tenso/commit/f66bd6f87bc50d6981c477683a33f9444ce073d3))

### Documentation

- Enhance README and index.html for clarity on performance and resource efficiency
  ([`9a7dd8f`](https://github.com/Khushiyant/tenso/commit/9a7dd8fcc049d725ea0d40dd8eaf365cc6b8c336))

### Features

- Add read_stream function and improve stream handling in core module
  ([`901d815`](https://github.com/Khushiyant/tenso/commit/901d815efa3d3301eab60b757829143e23c2ac25))

- Add support for complex data types and implement strict mode in serialization
  ([`c170f43`](https://github.com/Khushiyant/tenso/commit/c170f438c5ea05bd6eb07820dbb6e23c300890e4))

- Enhance README and benchmarks, update Tenso protocol to v2 with alignment support
  ([`baecc1d`](https://github.com/Khushiyant/tenso/commit/baecc1dd06467b1ad7aeab09bd5ed256b4af1d19))

- Enhance SEO and social media integration in index.html
  ([`0b208fb`](https://github.com/Khushiyant/tenso/commit/0b208fb0a1655f9540f751f88c4fd622ea29f041))

- Update Tenso protocol to v2 with alignment support, enhance serialization/deserialization, and add
  utility functions
  ([`a33ddd7`](https://github.com/Khushiyant/tenso/commit/a33ddd7fcb0bec08033d99957eefac5311073184))

- Update version to 0.4.3 and add tests for read_stream functionality
  ([`fdd2c66`](https://github.com/Khushiyant/tenso/commit/fdd2c666dd9fb7490b1db1354d0de5e122dc0573))

### Refactoring

- Replace manual tensor reception with core read_stream function in server
  ([`63d3544`](https://github.com/Khushiyant/tenso/commit/63d35448d5f2783d52dc5bf3334f546bc1d28666))

### Testing

- Refactor introspection tests for packet info retrieval and error handling
  ([`2b2d7e8`](https://github.com/Khushiyant/tenso/commit/2b2d7e86273652e87c5343489554a666ef28da4d))
