import scipy.stats as st
import numpy as np
import statsmodels.api as sm


def proportion_test(df_test, df_control, metric, alternative = 'larger', criteria = 0.05): 

    count = np.array([df_test[metric].sum(), df_control[metric].sum()]) 
    nobs = np.array([len(df_test), len(df_control)])

    z_stat, p_value = sm.stats.proportions_ztest(count, nobs, alternative = alternative)

    print(f'proportion test {df_test[metric].sum()/len(df_test)} \nproportion control {df_control[metric].sum()/len(df_control)}')

    print("p-value: ", p_value)

    if p_value < criteria:
        print(f"We reject the null hypothesis. \nThe proportion of the alternative is significantly {alternative}")
    else:
        print("We cannot reject the null hypothesis")

def ttest(df_test, df_control, metric, alternative = 'greater', criteria = 0.05):

    z_stat, p_value = st.ttest_ind(df_test[metric], df_control[metric], alternative=alternative)
    
    if p_value < criteria:
        print(f"We reject the null hypothesis. \nThe {metric} is significantly {alternative} in the test")
    else:
        print(f"We cannot reject the null hypothesis. \nThe {metric} is not significantly {alternative} in the test")
    
    print("test mean: ", df_test[metric].mean())
    print("control mean: ", df_control[metric].mean())
    print("p-value: ", p_value)