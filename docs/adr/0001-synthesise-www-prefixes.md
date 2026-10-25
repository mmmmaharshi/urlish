# Synthesise `www.` prefixes on negative rows

Ranked domain lists contain registrable domains only, so every negative row had
zero subdomains while real phishing rows often had one or more. The model learned
"has a subdomain = phishing" and scored `https://www.google.com` at 0.973. Rather
than drop the subdomain features, half the negatives get a `www.` prefix, which
breaks the shortcut at the cost of inventing URL forms that the source list did
not contain. The alternative — accepting the shortcut — was rejected because it
inflates every row of the report; the alternative of crawling real negative URLs
is the proper fix and is out of scope for a string-only benchmark.
