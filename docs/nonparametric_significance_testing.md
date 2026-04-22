# Nonparametric Significance Testing on Benchmark Results

First I collected evidence / references on how to use Wilcoxon signed-rank and Friedman test on our results.
None of the implemented papers name those explicitly. MoE-LoRA claim to use two-sided t-tests with p < 0.05, and MTL-LoRA use significance test at 90% confidence.

Another related paper we considered in the beginning is MoLA (https://aclanthology.org/2025.findings-naacl.284/). We did not implement this paper, since we implemented AdaMoLE. I remember that it was either newer or similar than MoLA which is why we did not choose MoLA.
MoLA states that they conducted a Friedman test to reject the null hypothesis that all algorithms perform equally. Following this they applied the Wilcoxon signed-rank test to perform pairwise comparisons between algorithms across six benchmarks to identify significant differences in performance. Then they applied FDR correction to adjust the p-values to address multiple comparisons.

I mostly copied this, since MoLA is a solid NAACL, peer reviewed findings paper.

### Significance test
1. We use Friedman as omnibus test over all approaches using alpha of 0.05. Each benchmark group is handled separately: Belebele, Taxi1500, SIB200, and FLORES split into bits_per_byte, byte_perplexity, and word_perplexity.
2. Then we use Wilcoxon signed-rank, two-sided, for pairwise comparisons within each benchmark group. Since we have multiple comparisons and many pairs we apply FDR correction so that some values are not only significant by chance.
3. Specifically we use Benjamini-Hochberg FDR for this.

### Implementation details
- The paired units are benchmark rows within each benchmark group.
- Friedman is run only on complete-case rows, meaning a row is used only if all approaches have a finite value.
- Wilcoxon is run on pairwise-complete rows, meaning a row is used if the two compared approaches both have finite values.
- Non-finite values such as `inf` or `nan` are excluded from the statistical tests.
- Pairwise Wilcoxon tests are only run if the Friedman test for that benchmark group is significant at alpha = 0.05.
- In the Friedman table, lower average rank means better.
