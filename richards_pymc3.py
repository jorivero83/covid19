import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
from epdata.parser import EpData
from datetime import datetime, timedelta
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 100)

if __name__ == '__main__':

    # Loading data
    country = 'Spain'
    oper_file_path1 = 'data/spain_data.csv'
    df_input = pd.read_csv(oper_file_path1, sep=';', decimal=',', error_bad_lines=False)
    df_input = EpData().parser(df=df_input)

    # Prepare data for training the model
    df = df_input[df_input['acumulado'] > 100][['date', 'acumulado', 'new_cases']].copy()
    df['days_since_100'] = list(np.arange(df.shape[0]))
    df.tail()

    # country = 'pe'
    # df_input = pd.read_csv('data/pe_data.csv', sep=';', decimal=',', error_bad_lines=False, parse_dates=['Date'])
    # df_input.columns = df_input.columns.map(lambda x: str(x).lower())
    # df_input.index = df_input.date
    # df_input['new_cases'] = df_input['confirmed'] - df_input['confirmed'].shift(1)
    #
    # # Prepare data for model training
    # df = df_input[df_input['confirmed'] > 100][['date', 'confirmed', 'new_cases']].copy()
    # df.columns = ['date', 'acumulado', 'new_cases']
    # df['days_since_100'] = list(np.arange(df.shape[0]))
    # print(df)

    # ========== Training the model =============
    x_values = df.days_since_100.values[:32]
    y_values = df.acumulado.astype('float64').values[:32]

    with pm.Model() as richards_model_final:
        sigma = pm.HalfCauchy('sigma', 1, shape=1)
        K = pm.Uniform('K', 100, 1000000, testval=5000)  # carrying capacity
        rate = pm.Normal('rate', 0.3, 0.05, testval=0.45)  # growth rate
        a = pm.Uniform('a', 0.1, 10.0, testval=0.25)
        x0 = pm.Uniform('x0', 1, 200, testval=15)
        T = pm.Uniform('T', 1, 3, testval=1.2)
        R0 = pm.Deterministic('R0', np.exp(rate * T))

        # Create likelihood for data
        x = pm.Data("x_data", x_values)
        acumulado = pm.Data("y_data", y_values)
        mu = pm.Deterministic('mu', K * (1 + np.exp(-rate * a * (x - x0 - (np.log(a) / (rate * a))))) ** (-1 / a))
        y = pm.Normal('y', mu=mu, tau=sigma, observed=acumulado)

        # Sample posterior
        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(4000, tune=4000, cores=7, start=start, target_accept=.95, random_seed=1234)

    # Fitted parameters
    f_values = (round(trace['rate'].mean(), 2), round(trace['x0'].mean(), 2),
                round(trace['K'].mean(), 2), round(trace['R0'].mean(), 2))
    txt = """
            --------------------------------------
            Fitted parameters for {}:
            --------------------------------------
            Growth rate: {}
            Turning point: {}
            Final size of epidemic: {}
            Basic reproduction number (R0): {}
            --------------------------------------
            """.format(country, *f_values)
    print(txt)

    os.makedirs('results/{}/'.format(country), exist_ok=True)
    az.plot_trace(trace, compact=True);
    plt.savefig('results/{}/trace_plot.png'.format(country))
    az.plot_posterior(trace);
    plt.savefig('results/{}/posterior_plot.png'.format(country))

    # ========== Compute predictions =============

    h = 4  # number points to prediction ahead
    with richards_model_final:
        # Update data so that we get predictions into the future
        x_data = np.arange(0, len(y_values) + h)
        y_data = np.array([np.nan] * len(x_data))
        pm.set_data({"x_data": x_data})
        pm.set_data({"y_data": y_data})

        # Sample posterior predictive
        post_pred_final = pm.sample_posterior_predictive(trace, samples=100)

    y_min_final = np.percentile(post_pred_final['y'], 2.5, axis=0)
    y_max_final = np.percentile(post_pred_final['y'], 97.5, axis=0)
    y_fit_final = np.percentile(post_pred_final['y'], 50, axis=0)
    dy_fit_final = trace['rate'].mean() * y_fit_final * (1 - (y_fit_final / trace['K'].mean()) ** trace['a'].mean())

    # Plot prediction of comulative cases
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plt.plot(df.days_since_100.values, df.acumulado.astype('float64').values,
             'b', marker='s', ls='-', lw=1, label='Observed data')
    plt.plot(np.arange(0, post_pred_final['y'].shape[1]), y_fit_final,
             'k', marker='^', ls=':', ms=5, mew=1, label='Model prediction')
    plt.fill_between(np.arange(0, post_pred_final['y'].shape[1]), y_min_final, y_max_final, color='0.5', alpha=0.5)
    ax.axvline(x_values[-1], ls='--', color='k', label='Day {} since 100 cases'.format(x_values[-1]))
    plt.text(x_values[-1], 2000,
             "{}".format(df[df.days_since_100 == x_values[-1]].date.map(lambda x: str(x)[:10]).values[0]),
             {'color': 'k', 'fontsize': 14},
             horizontalalignment='right', verticalalignment='baseline', rotation=90, clip_on=False)
    plt.suptitle('Richard model prediction: {}'.format(country), fontsize=24)
    f_values = (round(trace['rate'].mean(), 2), round(trace['x0'].mean(), 2),
                round(trace['K'].mean(), 2), round(trace['R0'].mean(), 2))
    plt.title(
        'Growth rate: {}, Turning point: {}, \n Final epidemic size: {}, Basic reproduction number: {}'.format(
            *f_values), fontsize=14)
    ax.set(xlabel='Days since 100 cases')
    ax.set(ylabel='Cumulative confirmed cases')
    plt.legend(loc='upper left')
    plt.savefig('results/{}/cumulative_prediction_plot.png'.format(country))

    # Incidence of new cases prediction
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plt.plot(df['days_since_100'], df['new_cases'], 'b', marker='s', ls='-', label='Observed data')
    plt.plot(np.arange(0, post_pred_final['y'].shape[1]), dy_fit_final, 'k', marker='^', ls=':', ms=5, mew=1,
             label='Predicted incidence of new cases')
    ax.axvline(x_values[-1], ls='--', color='k', label='Day {} since 100 cases'.format(x_values[-1]))
    plt.text(x_values[-1], 2000,
             "{}".format(df[df.days_since_100 == x_values[-1]].date.map(lambda x: str(x)[:10]).values[0]),
             {'color': 'k', 'fontsize': 14},
             horizontalalignment='right', verticalalignment='baseline', rotation=90, clip_on=False)
    plt.xlabel('Days since 100 cases')
    plt.ylabel('Incidence of new cases')
    plt.suptitle('Richard model prediction of incidence: {}'.format(country), fontsize=24)
    f_values = (round(trace['rate'].mean(), 2), round(trace['x0'].mean(), 2),
                round(trace['K'].mean(), 2), round(trace['R0'].mean(), 2))
    plt.title(
        'Growth rate: {}, Turning point: {}, \n Final epidemic size: {}, Basic reproduction number: {}'.format(
            *f_values), fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig('results/{}/newcases_prediction_plot.png'.format(country))

    # Save data with predictions
    df_pred = pd.DataFrame(index=pd.date_range(start=df.index[0], freq='D', periods=len(y_fit_final)))
    df_pred['days_since_100'] = np.arange(0, post_pred_final['y'].shape[1])
    df_pred['acumulado_pred'] = y_fit_final
    df_pred['acumulado_pred_lower'] = y_min_final
    df_pred['acumulado_pred_upper'] = y_max_final
    df_pred['new_cases_pred'] = dy_fit_final
    df_full = pd.merge(df_pred, df, on='days_since_100', how='left')
    df_full.index = df_pred.index
    df_full['date'] = df_pred.index
    cols_selected = ['date', 'days_since_100', 'new_cases', 'acumulado', 'acumulado_pred',
                     'acumulado_pred_lower', 'acumulado_pred_upper', 'new_cases_pred']
    dt_until = str(df[df.days_since_100 == x_values[-1]].date.map(lambda x: str(x)[:10]).values[0]).replace('-', '')
    df_full[cols_selected].to_csv('results/{}/{}_data_until_{}_predictions.csv'.format(country,country, dt_until),
                                  index=False, sep=',', decimal='.')
    print(df_full[cols_selected])
