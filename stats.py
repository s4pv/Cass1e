import warnings
from helper import Helper
from scipy import stats
from statsmodels.tsa import stattools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datapreparation import Datapreparation
import os
from outliers import smirnov_grubbs as grubbs

warnings.filterwarnings("ignore")

# Configuration and class variables
parsed_config = Helper.load_config('config.yml')

class Stats:

    def Shapiro_Wilk(dataset):
        try:
            df = Datapreparation.Reshape_Float(dataset)
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
        return stat, p

    def D_Agostino(dataset):
        try:
            df = Datapreparation.Reshape_Float(dataset)
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
        return stat, p

    def Anderson_Darling(dataset):
        try:
            df = Datapreparation.Reshape_Float(dataset)
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
        return sl, cv

    def Pearson(dataset1, dataset2):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset2)
            print("Pearson test. 2 Samples correlation test.")
            stat, p = stats.pearsonr(dataset1, dataset2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Spearman(dataset1, dataset2):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset2)
            print("Spearman test. 2 Samples correlation test.")
            stat, p = stats.spearmanr(dataset1, dataset2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Kendall(dataset1, dataset2):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset2)
            print("Kendall test. 2 Samples correlation test.")
            stat, p = stats.kendalltau(dataset1, dataset2)
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Chi_Squared(dataset1, dataset2):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset1)
            print("Chi^2 test. 2 Samples correlation test.")
            stat, p, dof, expected = stats.chi2_contingency(observed=[dataset1, dataset2])
            print('stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('Probably independent')
            else:
                print('Probably dependent')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Dickey_Fuller(dataset):
        try:
            df = Datapreparation.Reshape_Float(dataset)
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
        return stat, p

    def Kwiatkowski_Phillips_Schmidt_Shin(dataset):
        try:
            df = Datapreparation.Reshape_Float(dataset)
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
        return stat, p, lags, obs, crit, t

    def Student_T(dataset1, dataset2):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
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
        return stat, p

    def Paired_Student_T(dataset1, dataset2):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
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
        return stat, p

    def ANOVA(dataset1, dataset2):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
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
        return stat, p

    def Mann_Whitney(dataset1, dataset2):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
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
        return stat, p

    def Wilcoxon_Signed_Rank(dataset1, dataset2):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
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
        return stat, p

    def Kruskal_Wallis_H(dataset1, dataset2):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
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
        return stat, p

    def Friedman(dataset1, dataset2):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
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
        return stat, p


    def Plots(dataset):
        try:
            plt.hist(dataset)
            filename = os.path.join('stats_plots/histogram.png')
            plt.savefig(filename)
            plt.close()
            sm.ProbPlot(dataset).ppplot(line="r")
            filename = os.path.join('stats_plots/ppplot.png')
            plt.savefig(filename)
            plt.close()
            sm.ProbPlot(dataset).qqplot(line="r")
            filename = os.path.join('stats_plots/qqplot.png')
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Grubbs(dataset, coin):
        try:
            print("Outliers removal using the Grubbs test.")
            data_wo_outliers = grubbs.test(dataset, alpha=.05)
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return data_wo_outliers
