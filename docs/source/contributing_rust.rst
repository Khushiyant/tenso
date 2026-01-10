Contributing to Rust Code
=========================

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

.. code-block:: bash

    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    
    # Install Maturin
    pip install maturin
    
    # Clone and build
    git clone https://github.com/Khushiyant/tenso.git
    cd tenso
    maturin develop --release

Project Structure
~~~~~~~~~~~~~~~~~

.. code-block:: text

    src/
    ├── lib.rs                  # Main Rust module
    ├── tenso/                  # Python package
    │   ├── __init__.py
    │   ├── core.py            # Calls Rust functions
    │   └── tenso_rs.pyi       # Type stubs (optional)
    └── Cargo.toml             # Rust dependencies

Working on Rust Code
--------------------

Adding New Functions
~~~~~~~~~~~~~~~~~~~~

1. **Define the Rust function** in ``src/lib.rs``:

.. code-block:: rust

    #[pyfunction]
    fn my_new_function(py: Python, data: &PyArrayDyn<f32>) -> PyResult<Py<PyBytes>> {
        // Your implementation
        Ok(PyBytes::new(py, &result).into())
    }

2. **Export it in the module**:

.. code-block:: rust

    #[pymodule]
    fn tenso_rs(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(dumps_rs, m)?)?;
        m.add_function(wrap_pyfunction!(my_new_function, m)?)?;  // Add this
        Ok(())
    }

3. **Wrap it in Python** (``src/tenso/core.py``):

.. code-block:: python

    try:
        from .tenso_rs import my_new_function
        HAS_MY_FEATURE = True
    except ImportError:
        HAS_MY_FEATURE = False

    def my_feature(data):
        """User-facing docstring here."""
        if HAS_MY_FEATURE:
            return my_new_function(data)
        else:
            return fallback_implementation(data)

4. **Rebuild and test**:

.. code-block:: bash

    maturin develop --release
    pytest tests/test_core.py

Testing Changes
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run Rust tests
    cargo test
    
    # Run Python tests with new Rust code
    maturin develop && pytest -v
    
    # Benchmark performance
    python benchmark.py

Code Style
~~~~~~~~~~

- Follow Rust conventions (``cargo fmt``)
- Add doc comments for public functions:

.. code-block:: rust

    /// Serialize a NumPy array to Tenso format.
    ///
    /// # Arguments
    /// * `arr` - Input NumPy array (must be C-contiguous)
    /// * `check_integrity` - Whether to include XXH3 checksum
    ///
    /// # Returns
    /// Tenso packet as PyBytes
    #[pyfunction]
    fn dumps_rs(/* ... */) -> PyResult<Py<PyBytes>> {
        // ...
    }

Performance Guidelines
~~~~~~~~~~~~~~~~~~~~~~

- Use ``rayon`` for parallelism (already included)
- Avoid unnecessary allocations
- Use ``unsafe`` sparingly (document thoroughly)
- Profile with ``cargo flamegraph``

.. code-block:: bash

    # Install profiler
    cargo install flamegraph
    
    # Profile your code
    cargo build --release
    flamegraph -- python benchmark.py

Debugging
~~~~~~~~~

.. code-block:: bash

    # Build with debug symbols
    maturin develop
    
    # Use GDB/LLDB
    lldb python
    (lldb) run -c "import tenso; tenso.dumps(...)"

Common Issues
-------------

**Import Error: "cannot import name 'tenso_rs'"**
   Rebuild the extension: ``maturin develop --release``

**Segmentation Fault**
   Check array memory layout. Ensure C-contiguous arrays.

**Type Errors from PyO3**
   Verify NumPy dtype mapping in ``DType::from_code()``.

Resources
---------

- `PyO3 User Guide <https://pyo3.rs/>`_
- `Maturin Documentation <https://maturin.rs/>`_
- `Rust Book <https://doc.rust-lang.org/book/>`_
