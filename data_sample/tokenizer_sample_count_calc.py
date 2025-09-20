# Sample usage: bash data_sample/pylauncher.sh normal data_sample.tokenizer_sample_count_calc
import pandas as pd
import numpy as np
from data_sample.sample_generator_datatrove import _output_path

# Temperature parameter controls sampling distribution bias:
# Higher T (e.g., 5) creates more uniform sampling across subsets
# Lower T would increase bias toward larger subsets
T = 5                              # temperature
OUTPUT_PATH = _output_path('tokenizer_lang_sample_sizes.tsv')
INPUT_PATH = _output_path("sample_sizes.tsv")
TARGET = 10_000_000 # 10 million total samples
#TARGET = df['sample_size'].sum()   # keep original total

df = pd.read_csv(INPUT_PATH, sep='\t')

# Temperature-based weighting:
# Higher T: weights become more uniform (closer to 1.0)
# Lower T: weights emphasize larger sample sizes more strongly
df['weight'] = np.power(df['sample_size'], 1.0 / T)
df['prob']   = df['weight'] / df['weight'].sum()

raw = df['prob'] * TARGET

df['final_count'] = np.floor(raw).astype(int)

deficit = TARGET - df['final_count'].sum()
if deficit > 0:
    # fractional part of each raw count
    frac = raw - np.floor(raw)

    # indices of the `deficit` largest fractions
    top_idx = frac.nlargest(deficit).index

    # add +1 to those rows
    df.loc[top_idx, 'final_count'] += 1

# write the corrected column name
df[['subset','name','resource_category','sample_size','final_count']].to_csv(
    OUTPUT_PATH, sep='\t', index=False
)