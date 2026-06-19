/*
 * Tenso C ABI smoke test.
 *
 * Encodes a small f32 vector (Mode A: caller-allocates) and decodes it back
 * through the opaque view (Mode B: core-allocates), asserting the round trip and
 * exercising the null/length guards and the thread-local error channel.
 *
 * Build (from the workspace root), after `cargo build -p tenso-ffi`:
 *
 *   # static link:
 *   cc -I include \
 *      crates/tenso-ffi/examples/smoke.c \
 *      target/debug/libtenso_ffi.a \
 *      -lpthread -ldl -lm -o /tmp/tenso_smoke
 *   /tmp/tenso_smoke
 *
 *   # or dynamic link:
 *   cc -I include crates/tenso-ffi/examples/smoke.c \
 *      -L target/debug -ltenso_ffi -o /tmp/tenso_smoke
 *   LD_LIBRARY_PATH=target/debug /tmp/tenso_smoke    # (macOS: DYLD_LIBRARY_PATH)
 *
 * Exit code 0 == pass.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tenso.h"

static int failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s  (last_error=\"%s\")\n", (msg),         \
                    tenso_last_error());                                       \
            failures++;                                                        \
        }                                                                      \
    } while (0)

int main(void) {
    /* f32 [1,2,3,4], shape [4] */
    float elems[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t data[sizeof elems];
    memcpy(data, elems, sizeof elems);
    uint32_t shape[1] = {4};
    const uint8_t dtype_code = 1; /* f32 */

    /* --- Mode A: size then encode --- */
    uintptr_t needed = 0;
    int rc = tenso_dense_required_size(data, sizeof data, dtype_code, shape, 1,
                                       false, false, 0, &needed);
    CHECK(rc == TENSO_OK, "dense_required_size");
    CHECK(needed > 0, "needed > 0");

    uint8_t *packet = (uint8_t *)malloc(needed);
    CHECK(packet != NULL, "malloc packet");

    uintptr_t written = 0;
    rc = tenso_encode_dense_into(data, sizeof data, dtype_code, shape, 1, false,
                                 false, 0, packet, needed, &written);
    CHECK(rc == TENSO_OK, "encode_dense_into");
    CHECK(written > 0 && written <= needed, "written in range");

    /* --- parse_header --- */
    TensoHeader hdr;
    rc = tenso_parse_header(packet, written, &hdr);
    CHECK(rc == TENSO_OK, "parse_header");
    CHECK(hdr.version == 4, "version == 4");
    CHECK(hdr.dtype_code == dtype_code, "dtype_code");
    CHECK(hdr.ndim == 1, "ndim == 1");

    /* --- Mode B: decode opaque view --- */
    TensoView *view = tenso_decode(packet, written);
    CHECK(view != NULL, "tenso_decode");

    if (view) {
        CHECK(tenso_view_dtype(view) == dtype_code, "view dtype");
        CHECK(tenso_view_ndim(view) == 1, "view ndim");

        const uint32_t *vshape = tenso_view_shape(view);
        CHECK(vshape != NULL && vshape[0] == 4, "view shape[0] == 4");

        uintptr_t blen = tenso_view_body_len(view);
        CHECK(blen == sizeof data, "body len matches input");

        const uint8_t *bptr = tenso_view_body_ptr(view);
        CHECK(bptr != NULL && memcmp(bptr, data, sizeof data) == 0,
              "body round-trips");

        tenso_view_free(view);
    }

    free(packet);

    /* --- guard: null data with non-zero len --- */
    rc = tenso_parse_header(NULL, 8, &hdr);
    CHECK(rc == TENSO_ERR_NULL, "null data guard");

    /* --- guard: decode garbage returns NULL + sets error --- */
    uint8_t junk[4] = {0, 0, 0, 0};
    TensoView *bad = tenso_decode(junk, sizeof junk);
    CHECK(bad == NULL, "decode garbage -> NULL");
    CHECK(strlen(tenso_last_error()) > 0, "error message set");

    /* --- guard: null view accessors are total --- */
    CHECK(tenso_view_dtype(NULL) == 0, "null view dtype");
    CHECK(tenso_view_ndim(NULL) == 0, "null view ndim");
    CHECK(tenso_view_shape(NULL) == NULL, "null view shape");
    CHECK(tenso_view_body_ptr(NULL) == NULL, "null view body ptr");
    CHECK(tenso_view_body_len(NULL) == 0, "null view body len");
    tenso_view_free(NULL); /* no-op */

    if (failures == 0) {
        printf("tenso C ABI smoke test: OK\n");
        return 0;
    }
    fprintf(stderr, "tenso C ABI smoke test: %d failure(s)\n", failures);
    return 1;
}
