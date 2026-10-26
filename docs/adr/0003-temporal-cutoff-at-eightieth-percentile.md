# Temporal cutoff at the 80th percentile month

The temporal split trains on rows dated before a cutoff and tests on rows dated
on or after it. The cutoff is the first day of the month that contains the 80th
percentile row when all rows are ordered by collection date. It is computed from
the corpus rather than hard-coded, so a re-extraction with more months does not
silently date the split into the past.

Shared registrable domains are removed from the training half only, the same
asymmetry as ADR 0002: the test half holds the scarce positives, so discarding
there would throw away the measurement.

Rationale: a calendar cutoff (for example 2025-07-01) is more legible, but it
would have to be edited by hand for every new collection. Deriving it keeps the
split reproducible from the bytes, which is what the digest in the report
header promises. The value is printed in the report so a reader does not have
to run the code to find it.