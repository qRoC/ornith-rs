// The Q4_K oracle: ggml's own reference, dumped so `tools/q4k.py` can be
// compared against it byte for byte. `Q4K.md` step 1.
//
//   cmake -S ../vendor/ggml -B <build> -DGGML_METAL=OFF -DGGML_BLAS=OFF \
//         -DBUILD_SHARED_LIBS=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
//   cmake --build <build> -j8
//   cc -O2 -o q4k_oracle tools/q4k_oracle.c -L<build>/src -lggml-base \
//      -Wl,-rpath,<build>/src
//   ./q4k_oracle <n_elems> in.f32 blocks.bin deq.f32
//
// `q4k.py`'s reader must reproduce `deq.f32` at max_abs == 0 from `blocks.bin`.
// It does; that check covers the 12-byte scale packing, the nibble interleave
// and the subtraction, which are the three things a reimplementation gets wrong.
// The *writer* is expected to differ -- ggml's reference runs a search, ours is
// plain min/max and 0.77 points worse, which is why `q4k_ggml.py` calls ggml's
// quantizer rather than using `q4k.quantize` for anything that ships.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
void quantize_row_q4_K_ref(const float *x, void *y, int64_t k);
void dequantize_row_q4_K(const void *x, float *y, int64_t k);
int main(int argc, char **argv) {
    int64_t n = atoll(argv[1]);
    size_t nb = n / 256, bytes = nb * 144;
    float *x = malloc(n * 4), *r = malloc(n * 4);
    unsigned char *b = malloc(bytes);
    FILE *f = fopen(argv[2], "rb"); fread(x, 4, n, f); fclose(f);
    quantize_row_q4_K_ref(x, b, n);
    dequantize_row_q4_K(b, r, n);
    f = fopen(argv[3], "wb"); fwrite(b, 1, bytes, f); fclose(f);
    f = fopen(argv[4], "wb"); fwrite(r, 4, n, f); fclose(f);
    printf("ok: %lld elems, %zu blocks, %zu bytes\n", (long long)n, nb, bytes);
    return 0;
}
