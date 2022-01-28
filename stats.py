import warnings

from helper import Helper
from scipy import stats
from statsmodels.tsa import stattools
import numpy
from dataprocessing import DataProcessing

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

class Stats:

    def Shapiro_Wilk(dataset):
        try:
            df = DataProcessing.Reshape_Data(dataset, 'no model')
            print('Shapiro Wilk normality test')
            stat, p = stats.shapiro(df)
            print('stat=%.3f,p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably Gaussian')
            else:
                print('Probably not Gaussian')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def D_Agostino(dataset):
        try:
            df = DataProcessing.Reshape_Data(dataset, 'no model')
            print("D'Agostino normality test")
            stat, p = stats.normaltest(df)
            print('stat=%.3f,p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably Gaussian')
            else:
                print('Probably not Gaussian')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Anderson_Darling(dataset):
        try:
            df = DataProcessing.Reshape_Data(dataset, 'no model')
            print("Anderson Darling normality test")
            results = stats.anderson(df)
            print('stat=%.3f' % (results.statistic))
            for i in range(len(results.critical_values)):
                sl, cv = results.significance_level[i], results.critical_values[i]
                if results.statistic < cv:
                    print('Probably Gaussian at the %.1f%% level' % (sl))
                else:
                    print('Probably not Gaussian at the %.1f%% level' % (sl))
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Pearson(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Pearson test. 2 Samples correlation test.")
            stat, p = stats.pearsonr(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Spearman(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Spearman test. 2 Samples correlation test.")
            stat, p = stats.spearmanr(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Kendall(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Kendall test. 2 Samples correlation test.")
            stat, p = stats.kendalltau(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Chi_Squared(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Chi^2 test. 2 Samples correlation test.")
            stat, p = stats.chi2(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Dickey_Fuller(dataset):
        try:
            df = DataProcessing.Reshape_Data(dataset, 'no model')
            print("Dickey Fuller test. testing for unit root.")
            stat, p, lags, obs, crit, t = stattools.adfuller(df)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably not Stationary')
            else:
                print('Probably Stationary')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Kwiatkowski_Phillips_Schmidt_Shin(dataset):
        try:
            df = DataProcessing.Reshape_Data(dataset, 'no model')
            print("Kwiatkowski-Phillips-Schmidt-Shin test. testing for unit root.")
            stat, p, lags, obs, crit, t = stattools.kpss(df)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably Stationary')
            else:
                print('Probably not Stationary')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Student_T(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Student's t-test. testing for identity in 2 samples distributions.")
            stat, p = stats.ttest_ind(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably the same distribution')
            else:
                print('Probably different distributions')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Paired_Student_T(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Student's t-test. testing for identity in 2 samples distributions.")
            stat, p = stats.ttest_rel(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably the same distribution')
            else:
                print('Probably different distributions')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def ANOVA(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("ANOVA test. testing for identity in 2 samples distributions.")
            stat, p = stats.f_oneway(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably the same distribution')
            else:
                print('Probably different distributions')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Mann_Whitney(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Mann-Whitney U test. testing for identity in 2 samples distributions.")
            stat, p = stats.mannwhitneyu(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably the same distribution')
            else:
                print('Probably different distributions')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Wilcoxon_Signed_Rank(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Wilcoxon Signed-Rank test. testing for identity in 2 samples distributions.")
            stat, p = stats.wilcoxon(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably the same distribution')
            else:
                print('Probably different distributions')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Kruskal_Wallis_H(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Kruskal-Wallis H test. testing for identity in 2 samples distributions.")
            stat, p = stats.kruskal(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably the same distribution')
            else:
                print('Probably different distributions')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Friedman(dataset1, dataset2):
        try:
            df1 = DataProcessing.Reshape_Data(dataset1, 'no model')
            df2 = DataProcessing.Reshape_Data(dataset2, 'no model')
            print("Friedman test. testing for identity in 2 samples distributions.")
            stat, p = stats.friedmanchisquare(df1, df2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably the same distribution')
            else:
                print('Probably different distributions')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
