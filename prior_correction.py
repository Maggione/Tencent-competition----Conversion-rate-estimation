import pandas as pd
import sys

y_prob = float(sys.argv[1])
infile = sys.argv[2]
outfile = sys.argv[3]

df = pd.read_csv(infile)
tau = 0.0272320301462

def correct(df):
    # y = train["label"].mean()
    y = y_prob
    apha = tau / y
    beta = (1 - tau) / (1 - y)
    print apha, beta

    def f(p):
        p = apha * p / (apha * p + beta *(1 - p)) 
        return p

    df_correct = df.copy()
    df_correct["prob"] = df_correct["prob"].apply(lambda x : f(x))

    y = df_correct["prob"].mean()
    apha = tau / y
    print apha

    return df_correct

test_correct = correct(df)
test_correct.to_csv(outfile, index = False)
