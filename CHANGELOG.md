# CHANGELOG


## v0.10.0 (2025-12-26)

### Chores

- Add vectored Tenso benchmark for zero-copy transmission; improve size reporting in serialization
  benchmarks
  ([`cf5a67e`](https://github.com/Khushiyant/tenso/commit/cf5a67efcee2cea0d5b3ac69a5c92dea5580848a))

- Refactor and enhance Tenso library with improved error handling, code formatting, and test
  coverage
  ([`a0ffc18`](https://github.com/Khushiyant/tenso/commit/a0ffc18bc46ea0d00118bed90761461e542e6800))

- Refactor Tenso serialization and streaming for improved clarity and performance
  ([`d5d022b`](https://github.com/Khushiyant/tenso/commit/d5d022bddd379a5545c966cb2a8f7833ce7c7657))

- Enhanced docstrings across modules for better parameter and return type descriptions. -
  Streamlined the `aread_stream` and `awrite_stream` functions to reduce redundancy and improve
  readability. - Added support for sparse tensors in the `dumps` and `loads` functions. - Improved
  integrity checks and error handling in the serialization process. - Updated FastAPI integration to
  handle tensor streaming more efficiently. - Introduced GPU support for direct serialization from
  device memory. - Optimized memory allocation and data transfer methods for better performance. -
  General code cleanup and organization for maintainability.

### Features

- Enhance Tenso with new features including LZ4 compression, sparse matrix support, and multi-tensor
  bundling; add comprehensive tests for advanced functionalities
  ([`68c90b9`](https://github.com/Khushiyant/tenso/commit/68c90b9676fd03c7427e98b0a589c88d5748a104))


## v0.9.1 (2025-12-22)

### Bug Fixes

- Improve write_stream function by removing os.writev optimization to prevent data corruption;
  enhance TensoResponse for better tensor streaming with custom headers and error handling
  ([`2262d4a`](https://github.com/Khushiyant/tenso/commit/2262d4a09d8a8bf474a096d5bc6ff173941c1f8f))


## v0.9.0 (2025-12-22)

### Bug Fixes

- Add XXH3 integrity checks; update README and benchmark for integrity support; adjust
  pyproject.toml dependencies
  ([`b8612b0`](https://github.com/Khushiyant/tenso/commit/b8612b0d8cc28c841ef6cc9491c35f58b3353177))

### Features

- Enhance Tenso with DoS protection, FastAPI integration, and GPU support
  ([`e0f3d2a`](https://github.com/Khushiyant/tenso/commit/e0f3d2ae99939041c179557f8cea2b531e4ee224))

- Added security limits for maximum dimensions (MAX_NDIM) and maximum elements (MAX_ELEMENTS) to
  prevent allocation attacks. - Improved error handling in core functions to raise ValueErrors for
  packets exceeding security limits. - Introduced FastAPI integration with a new TensoResponse class
  for efficient tensor streaming. - Enhanced GPU support by adding JAX as a backend option alongside
  CuPy and PyTorch. - Implemented async read/write functions for better performance in asynchronous
  environments. - Added comprehensive tests for new features, including DoS protection, FastAPI
  response handling, and GPU device reading. - Updated documentation and comments for clarity and
  maintainability.


## v0.8.0 (2025-12-21)

### Features

- Add xxhash for integrity checks; enhance async read and write functions; update utils for packet
  metadata extraction
  ([`258ce17`](https://github.com/Khushiyant/tenso/commit/258ce1718e4cc00f85085fd9b8a6dbfde6bd720a))


## v0.7.0 (2025-12-21)

### Features

- Add iter_dumps function for zero-copy serialization; optimize read_stream and write_stream for
  efficiency
  ([`8956566`](https://github.com/Khushiyant/tenso/commit/89565661d7742dc7c7a4831a78b7791ce62c56ca))


## v0.6.1 (2025-12-18)

### Bug Fixes

- Correct tensor view and reshape logic in read_to_device function; improve mock assertions in tests
  ([`55ecd84`](https://github.com/Khushiyant/tenso/commit/55ecd84430b9ca5d8c528241730692b7c49d5bea))

### Chores

- Add scripts for documentation generation and benchmarking
  ([`74f4bc9`](https://github.com/Khushiyant/tenso/commit/74f4bc91317a4c91bc066a7118fca50cf5d7461f))

- Enhance project metadata retrieval in Sphinx configuration and improve backend detection logic
  ([`dc294d5`](https://github.com/Khushiyant/tenso/commit/dc294d53ba3560d4cebf7eabaab0d32cc8722e79))

- Update GitHub Actions workflow for documentation build and deployment
  ([`f555a65`](https://github.com/Khushiyant/tenso/commit/f555a65cc3988113289e8bbbeb4953c743222398))

- Update README and documentation scripts; remove unused scripts and add mock imports for
  dependencies
  ([`3d5fc9c`](https://github.com/Khushiyant/tenso/commit/3d5fc9c2aa173bef54272338eda426573468a868))


## v0.6.0 (2025-12-16)

### Chores

- Add GitHub Actions workflow for documentation generation and deployment
  ([`8ab1c43`](https://github.com/Khushiyant/tenso/commit/8ab1c43f0ca36c816dd313bb253902153405289f))

### Documentation

- Add Sphinx documentation makefiles and update API references
  ([`3216226`](https://github.com/Khushiyant/tenso/commit/3216226b536b07545d87f55d207b0c412f46e648))

### Features

- Add GPU support with read_to_device function and corresponding tests
  ([`31bed37`](https://github.com/Khushiyant/tenso/commit/31bed371e86d55635c4db8ac49eed824204703c0))

### Refactoring

- Remove benchmark_io.py and update pyproject.toml for GPU dependencies
  ([`2f06485`](https://github.com/Khushiyant/tenso/commit/2f064850fd5d015814fc5e2d76b0f089a9aa72d9))


## v0.5.1 (2025-12-15)

### Bug Fixes

- Correct section header for optional dependencies in pyproject.toml
  ([`8ec6743`](https://github.com/Khushiyant/tenso/commit/8ec6743cce35b0bb474fec75c93aadf9de823179))


## v0.5.0 (2025-12-15)

### Chores

- Add resource monitoring benchmarks and enhance argument parser for comprehensive testing
  ([`a42c700`](https://github.com/Khushiyant/tenso/commit/a42c7006baf4693622ec91127c17ac6ff7a6f699))

### Documentation

- Enhance README.md with detailed benchmarks, usage examples, and performance comparisons
  ([`09967f5`](https://github.com/Khushiyant/tenso/commit/09967f579daec9af3f779839bb7dc5a3c29db92d))

- Update README.md with enhanced benchmark scenarios and usage examples for network streaming and
  serialization
  ([`6d41b22`](https://github.com/Khushiyant/tenso/commit/6d41b22174d0e7869631713f615597c71a489913))

### Features

- Implement async zero-copy read stream and enhance core serialization methods
  ([`9a4ce40`](https://github.com/Khushiyant/tenso/commit/9a4ce40da239c9e31d69d02dac48b9053dba2ed7))


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
