from sklearn.linear_model import LogisticRegression
from causallib.estimation import IPW
from causallib.datasets import load_nhefs
# %matplotlib inline
import matplotlib.pyplot as plt

from causallib.evaluation import PropensityEvaluator
import matplotlib


def main():
    data = load_nhefs()
    learner = LogisticRegression(penalty='none',  # No regularization, new in scikit-learn 0.21.*
                                 solver='lbfgs',
                                 max_iter=1000)  # Increaed to achieve convergence with 'lbfgs' solver
    ipw = IPW(learner)
    ipw.fit(data.X, data.a)
    potential_outcomes = ipw.estimate_population_outcome(data.X, data.a, data.y)
    effect = ipw.estimate_effect(potential_outcomes[1], potential_outcomes[0])
    print(effect)  # a

    evaluator = PropensityEvaluator(ipw)
    evaluation_results = evaluator.evaluate_simple(data.X, data.a, data.y, plots=["covariate_balance_love"])
    fig = evaluation_results.plots["covariate_balance_love"].get_figure()
    fig.set_size_inches(6, 6)  # set a more compact size than default
    # fig.show()

    import pandas as pd
    def group_and_observe_by(var, k, plot=True):
        step_size = (data.X[var].max() - data.X[var].min())/k
        step_size = int(round(step_size))
        groups = range(data.X[var].min(), data.X[var].max() + step_size, step_size)
        print(list(groups))
        grouped = pd.cut(data.X[var], bins=groups, include_lowest=True)
        grouped.head()

        observed_diff = data.y.groupby([data.a, grouped]).mean()
        observed_diff = observed_diff.xs(1) - observed_diff.xs(0)
        observed_diff = observed_diff.rename("observed_effect")
        if plot:
            ax = observed_diff.plot(kind="barh")
            ax.set_xlabel(var)
            ax.set_xlabel("observed_effect")
            plt.show()

        frequency = grouped.value_counts(sort=False)
        frequency.rename("counts", inplace=True)
        if plot:
            ax = frequency.plot(kind="barh")
            ax.set_xlabel("counts")
            plt.show()

        by_var = observed_diff.to_frame().join(frequency)
        tx_distribution = data.a.groupby(grouped, observed=True).value_counts(dropna=False)
        tx_distribution = tx_distribution.unstack("qsmk")
        tx_distribution["propensity"] = tx_distribution[1] / tx_distribution.sum(axis="columns")

        by_var = by_var.join(tx_distribution)

        print(by_var)

        import numpy as np
        # drop nan(s)
        masked_data = np.ma.masked_array(by_var["observed_effect"], np.isnan(by_var["observed_effect"]))
        avg = np.ma.average(masked_data, weights=by_var["counts"])
        print(avg)

        return avg

    # avg = group_and_observe_by("age", k=10, plot=False)
    # group_and_observe_by("smokeintensity", k=10)

    def affect_by_k_groups(var):
        plt.close()
        affects = []
        xs = range(1, 21)
        for i in xs:
            affects.append(group_and_observe_by(var, k=i, plot=False))

        print(affects)
        plt.plot(xs, affects)
        plt.show()



    affect_by_k_groups("age")




if __name__ == '__main__':
    main()  # part 3
