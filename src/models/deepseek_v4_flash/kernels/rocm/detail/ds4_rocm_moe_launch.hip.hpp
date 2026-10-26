/* Dynamic LDS for the Q2-down WMMA kernel. The epilogue aliases its
 * float C staging over the half A/B staging buffers, so allocation needs the
 * larger region rather than their sum. */
static size_t ds4_rocm_q2_down_wmma_shmem(uint32_t mtiles, uint32_t bm,
                                          uint32_t bn, uint32_t bk) {
    const size_t ab = ((size_t)mtiles * bm * bk + 2u * (size_t)bk * bn) *
                      sizeof(__half);
    const size_t c = (2u * (size_t)mtiles * bm * bn) * sizeof(float);
    return ab > c ? ab : c;
}
