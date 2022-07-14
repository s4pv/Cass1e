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

    def Shapiro_Wilk(dataset, coin, type, date):
        try:
            df = Datapreparation.Reshape_Float(dataset)
            a = 'Shapiro Wilk normality test'
            print(a)
            stat, p = stats.shapiro(df)
            b = 'stat=%.3f,p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably Gaussian'
                print(c)
            else:
                c = 'Probably not Gaussian'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Shapiro_Wilk_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def D_Agostino(dataset, coin, type, date):
        try:
            df = Datapreparation.Reshape_Float(dataset)
            a = "D'Agostino normality test"
            print(a)
            stat, p = stats.normaltest(df)
            b = 'stat=%.3f,p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably Gaussian'
                print(c)
            else:
                c = 'Probably not Gaussian'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_D_Agostino_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Anderson_Darling(dataset, coin, type, date):
        try:
            df = Datapreparation.Reshape_Float(dataset)
            a = 'Anderson Darling normality test'
            print(a)
            results = stats.anderson(df)
            b = 'stat=%.3f' % (results.statistic)
            print(b)
            for i in range(len(results.critical_values)):
                sl, cv = results.significance_level[i], results.critical_values[i]
                if results.statistic < cv:
                    c = 'Probably Gaussian at the %.1f%% level' % (sl)
                    print(c)
                else:
                    c = 'Probably not Gaussian at the %.1f%% level' % (sl)
                    print(c)
                lines = [str(a), str(b), str(c)]
                if type == 'model':
                    filedir = 'model_stats/' + str(date) + '/'
                elif type == 'forecast':
                    filedir = 'forecast_stats/' + str(date) + '/'
                os.makedirs(filedir, exist_ok=True)
                filename = os.path.join(filedir, str(coin['symbol']) + '_Anderson_Darling_stats.txt')
                with open(filename, 'w') as f:
                    f.write('\n'.join(lines))
                    f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return sl, cv

    def Pearson(dataset1, dataset2, coin, type, date):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'Pearson test. 2 Samples correlation test.'
            print(a)
            stat, p = stats.pearsonr(dataset1, dataset2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably independent'
                print(c)
            else:
                c = 'Probably dependent'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Pearson_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Spearman(dataset1, dataset2, coin, type, date):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'Spearman test. 2 Samples correlation test.'
            print(a)
            stat, p = stats.spearmanr(dataset1, dataset2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably independent'
                print(c)
            else:
                c = 'Probably dependent'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Spearman_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Kendall(dataset1, dataset2, coin, type, date):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'Kendall test. 2 Samples correlation test.'
            print(a)
            stat, p = stats.kendalltau(dataset1, dataset2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably independent'
                print(c)
            else:
                c = 'Probably dependent'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Kendall_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Chi_Squared(dataset1, dataset2, coin, type, date):
        try:
            #df1 = Datapreparation.Reshape_Float(dataset1)
            #df2 = Datapreparation.Reshape_Float(dataset1)
            a = 'Chi^2 test. 2 Samples correlation test.'
            print(a)
            stat, p, dof, expected = stats.chi2_contingency(observed=[dataset1, dataset2])
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably independent'
                print(c)
            else:
                c = 'Probably dependent'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Chi_Squared_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Dickey_Fuller(dataset, coin, type, date):
        try:
            df = Datapreparation.Reshape_Float(dataset)
            a = 'Dickey Fuller test. testing for unit root.'
            print(a)
            stat, p, lags, obs, crit, t = stattools.adfuller(df)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably not Stationary'
                print(c)
            else:
                c = 'Probably Stationary'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Dickey_Fuller_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Kwiatkowski_Phillips_Schmidt_Shin(dataset, coin, type, date):
        try:
            df = Datapreparation.Reshape_Float(dataset)
            a = 'Kwiatkowski-Phillips-Schmidt-Shin test. testing for unit root.'
            print(a)
            stat, p, lags, obs, crit, t = stattools.kpss(df)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably Stationary'
                print(c)
            else:
                c = 'Probably not Stationary'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Kwiatkowski_Phillips_Schmidt_Shin_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p, lags, obs, crit, t

    def Student_T(dataset1, dataset2, coin, type, date):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
            a = "Student's t-test. testing for identity in 2 samples distributions."
            print(a)
            stat, p = stats.ttest_ind(df1, df2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably the same distribution'
                print(c)
            else:
                c = 'Probably different distributions'
                print()
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Student_T_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Paired_Student_T(dataset1, dataset2, coin, type, date):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
            a = "Student's t-test. testing for identity in 2 samples distributions."
            print(a)
            stat, p = stats.ttest_rel(df1, df2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably the same distribution'
                print(c)
            else:
                c = 'Probably different distributions'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Paired_Student_T_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def ANOVA(dataset1, dataset2, coin, type, date):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'ANOVA test. testing for identity in 2 samples distributions.'
            print(a)
            stat, p = stats.f_oneway(df1, df2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably the same distribution'
                print(c)
            else:
                c = 'Probably different distributions'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_ANOVA_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Mann_Whitney(dataset1, dataset2, coin, type, date):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'Mann-Whitney U test. testing for identity in 2 samples distributions.'
            print(a)
            stat, p = stats.mannwhitneyu(df1, df2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably the same distribution'
                print(c)
            else:
                c = 'Probably different distributions'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Mann_Whitney_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Wilcoxon_Signed_Rank(dataset1, dataset2, coin, type, date):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'Wilcoxon Signed-Rank test. testing for identity in 2 samples distributions.'
            print(a)
            stat, p = stats.wilcoxon(df1, df2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably the same distribution'
                print(c)
            else:
                c = 'Probably different distributions'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Wilcoxon_Signed_Rank_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Kruskal_Wallis_H(dataset1, dataset2, coin, type, date):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'Kruskal-Wallis H test. testing for identity in 2 samples distributions.'
            print(a)
            stat, p = stats.kruskal(df1, df2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably the same distribution'
                print(c)
            else:
                c = 'Probably different distributions'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Kruskal_Wallis_H_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p

    def Friedman(dataset1, dataset2, coin, type, date):
        try:
            df1 = Datapreparation.Reshape_Float(dataset1)
            df2 = Datapreparation.Reshape_Float(dataset2)
            a = 'Friedman test. testing for identity in 2 samples distributions.'
            print(a)
            stat, p = stats.friedmanchisquare(df1, df2)
            b = 'stat=%.3f, p=%.3f' % (stat, p)
            print(b)
            if p > 0.05:
                c = 'Probably the same distribution'
                print(c)
            else:
                c = 'Probably different distributions'
                print(c)
            lines = [str(a), str(b), str(c)]
            if type == 'model':
                filedir = 'model_stats/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_Friedman_stats.txt')
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
                f.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return stat, p


    def Plots(dataset, coin, type, date):
        try:
            plt.hist(dataset)
            if type == 'model':
                filedir = 'model_stats_plots/' + str(date) + '/'
            elif type == 'forecast':
                filedir = 'forecast_stats_plots/' + str(date) + '/'
            os.makedirs(filedir, exist_ok=True)
            filename = os.path.join(filedir, str(coin['symbol']) + '_histogram.png')
            plt.savefig(filename)
            plt.close()
            sm.ProbPlot(dataset).ppplot(line="r")
            filename = os.path.join(filedir, str(coin['symbol']) + '_ppplot.png')
            plt.savefig(filename)
            plt.close()
            sm.ProbPlot(dataset).qqplot(line="r")
            filename = os.path.join(filedir, str(coin['symbol']) + '_qqplot.png')
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Grubbs(dataset):
        try:
            print('Outliers removal using the Grubbs test.')
            data_wo_outliers = grubbs.test(dataset, alpha=.05)
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return data_wo_outliers
