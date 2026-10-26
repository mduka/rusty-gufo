/* Non-causal block attention for the DSpark drafter.
 *
 * The drafted block is produced in one pass, so every block row must see the
 * injected target KV *and* every other block row. That is the opposite of the
 * target's causal window, hence a dedicated launcher rather than a mask
 * variation on the causal kernels. */
