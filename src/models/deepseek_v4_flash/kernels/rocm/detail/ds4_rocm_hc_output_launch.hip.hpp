/* The vec4 expand kernels read and write four adjacent embedding lanes as one
 * 16-byte access, which needs the row pitch to be a multiple of four and every
 * base 16-byte aligned. hipMalloc over-aligns, but these tensors can be views
 * into a larger arena, so check rather than assume. */
static inline bool hip_hc_vec4_ok(uint32_t n_embd, const void *a, const void *b,
                                  const void *c) {
    return (n_embd & 3u) == 0u &&
           (((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 15u) == 0u;
}

