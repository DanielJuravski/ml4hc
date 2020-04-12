import lifelines as ll
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from lifelines.statistics import logrank_test
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

RETINOPATHY_DATA = '/home/daniel/Documents/MachineLearning4HealthCare/ml4hc/data/diabetic_retinopathy/data.xlsx'
GBM_DATA = '/home/daniel/Documents/MachineLearning4HealthCare/ml4hc/data/glioblastomamutations/data.csv'


def rt_task():
    rt = pd.read_excel(RETINOPATHY_DATA)
    print(rt.head())

    tr_km = ll.KaplanMeierFitter()
    tr_km_plt = tr_km.fit(rt["tr_time"], rt["tr_status"], label="Treated KM estimate").plot()

    untr_km = ll.KaplanMeierFitter()
    untr_km_plt = untr_km.fit(rt["untr_time"], rt["untr_status"], label="Untreated KM estimate").plot()

    plt.title("Treated VS Untreated KM estimates")
    plt.show()

    log_rank_test = ll.statistics.logrank_test(rt["tr_time"], rt["untr_time"],
                                               rt["tr_status"], rt["untr_status"])
    log_rank_test.print_summary()

    lt1_samples = rt.query("laser_type==1")
    lt2_samples = rt.query("laser_type==2")

    lt1_km = ll.KaplanMeierFitter()
    lt1_km_plt = lt1_km.fit(lt1_samples["tr_time"], lt1_samples["tr_status"], label="Xenon laser").plot()

    lt2_km = ll.KaplanMeierFitter()
    lt2_km_plt = lt2_km.fit(lt2_samples["tr_time"], lt2_samples["tr_status"], label="Argon laser").plot()

    plt.title("Xenon VS Argon KM estimates")
    plt.show()

    log_rank_test = ll.statistics.logrank_test(lt1_samples["tr_time"], lt2_samples["tr_time"],
                                               lt1_samples["tr_status"], lt2_samples["tr_status"])
    log_rank_test.print_summary()


def encode_life_race(gbm):
    le_life = LabelEncoder()
    gbm['e_Life'] = le_life.fit_transform(gbm['Life']) + 1
    ohe = OneHotEncoder()
    race = ohe.fit_transform(gbm['Race'].values.reshape(-1, 1)).toarray()
    dfOneHot = pd.DataFrame(race, columns=["Race_" + str(int(i)) for i in range(race.shape[1])])
    gbm = pd.concat([gbm, dfOneHot], axis=1)

    return gbm


def find_significant_genes(gbm, G_START, G_END):
    gene2pval = {}
    for g in gbm.columns[G_START:G_END + 1]:
        g_is_0_samples = gbm.query("{0}==0".format(g))
        g_is_1_samples = gbm.query("{0}==1".format(g))
        log_rank_test = ll.statistics.logrank_test(g_is_0_samples["Days Till Death"], g_is_1_samples["Days Till Death"],
                                                   g_is_0_samples['e_Life'], g_is_1_samples['e_Life'])
        gene2pval[g] = log_rank_test.p_value

    # sort by p-vals
    gene2pval = {k: v for k, v in sorted(gene2pval.items(), key=lambda item: item[1])}

    num_significant_g = 0
    for g, p in gene2pval.items():
        if p < 0.05:
            num_significant_g += 1
            print("Significant gene: {0} p-val: {1}".format(g, p))
    print("Number significant genes: {0}".format(num_significant_g))

    return gene2pval


def find_fdr_group(gene2pval, G_NUM):
    print("Genes in FDR group:")
    for i, (g, p) in enumerate(gene2pval.items()):
        fdr = 0.05 * (i + 1) / G_NUM
        if p < fdr:
            print("Significant gene: {0} p-val: {1} fdr-val: {2}".format(g, p, fdr))


def train_cox_model(data, features):
    gbm_mini = data[features]
    gbm_cox = ll.CoxPHFitter()
    gbm_cox.fit(gbm_mini, duration_col='Days Till Death', event_col='e_Life')
    gbm_cox.print_summary()

    return gbm_cox.log_likelihood_, gbm_cox.concordance_index_

def gbm_task():
    G_START = 6
    G_END = 505
    G_NUM = G_END - G_START + 1

    gbm = pd.read_csv(GBM_DATA)
    gbm = encode_life_race(gbm)

    gene2pval = find_significant_genes(gbm, G_START, G_END)

    find_fdr_group(gene2pval, G_NUM)

    # COX models
    models_info_list = []
    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "Race_0", "Race_1", "Race_2", "NLRP4", "ITGAD", "IDH1", "CALN1", "MAGI1", "ITGB4"])
    models_info_list.append({"model features": "Race, NLRP4, ITGAD, IDH1, CALN1, MAGI1, ITGB4",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "NLRP4", "ITGAD", "IDH1", "CALN1", "MAGI1", "ITGB4"])
    models_info_list.append({"model features": "NLRP4, ITGAD, IDH1, CALN1, MAGI1, ITGB4",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "NLRP4"])
    models_info_list.append({"model features": "NLRP4",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "ITGAD"])
    models_info_list.append({"model features": "ITGAD",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "IDH1"])
    models_info_list.append({"model features": "IDH1",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "CALN1"])
    models_info_list.append({"model features": "CALN1",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "MAGI1"])
    models_info_list.append({"model features": "MAGI1",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    log_liklihood, concordance = train_cox_model(gbm, ["Days Till Death", "e_Life", "ITGB4"])
    models_info_list.append({"model features": "ITGB4",
                             "log likelihood": log_liklihood,
                             "concordance": concordance})

    models_info = pd.DataFrame(models_info_list)
    print(models_info)


if __name__ == '__main__':
    rt_task()
    gbm_task()


