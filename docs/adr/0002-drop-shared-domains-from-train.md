# Drop shared domains from the training half of a collection split

Majestic and Tranco overlap heavily, and so do Phishing Army and OpenPhish, so a
collection split shares thousands of registrable domains across the boundary and
measures leakage instead of transfer. Shared domains are removed from the
training half only. Removing them from both halves is symmetric and looks fairer,
but it deletes most of the 300 OpenPhish positives and leaves a test set too small
to estimate recall, so the measurement takes priority over the symmetry. The
consequence to remember: training rows are dropped without being replaced, which
is why `n fit` is 12,756 for that row while the other rows train on 43,380.
