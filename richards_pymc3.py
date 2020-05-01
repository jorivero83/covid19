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
import plotly.offline as py
import plotly.graph_objs as go

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 100)

if __name__ == '__main__':

    epidemic_start = 100
    x_data_label = 'days_since_{}'.format(epidemic_start)

    # Loading data
    country = 'Spain'
    oper_file_path1 = 'data/spain_data.csv'
    df_input = pd.read_csv(oper_file_path1, sep=';', decimal=',', error_bad_lines=False)
    df_input = EpData().parser(df=df_input)

    # Prepare data for training the model
    df = df_input[df_input['acumulado'] > epidemic_start][['date', 'acumulado', 'new_cases']].copy()
    df[x_data_label] = list(np.arange(df.shape[0]))
    print(df.tail(10))
    ndays_limit = 58

    # country = 'Cuba'
    # oper_file_path1 = 'data/cuba_data.csv'
    # df_input = pd.read_csv(oper_file_path1, sep=',', decimal='.', error_bad_lines=False, parse_dates=['date'])
    # df_input.index = df_input['date']
    #
    # # Prepare data for model training
    # df = df_input[df_input['confirmed'] > epidemic_start][['date', 'confirmed', 'new_cases']].copy()
    # df.columns = ['date', 'acumulado', 'new_cases']
    # df[x_data_label] = list(np.arange(df.shape[0]))
    # print(df.tail(10))
    # ndays_limit = 31

    # ========== Training the model =============
    x_values = df[x_data_label].values[:ndays_limit]
    y_values = df.acumulado.astype('float64').values[:ndays_limit]
    dt_until = str(df[df[x_data_label] == x_values[-1]].date.map(lambda x: str(x)[:10]).values[0]).replace('-', '')
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

    os.makedirs('results/{}/'.format(country), exist_ok=True)
    with richards_model_final:

        # Keep model
        db = pm.backends.Text('results/{}/{}_model_{}'.format(country,country,dt_until))

        # Sample posterior
        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(4000, tune=4000, cores=7, start=start, target_accept=.95,
                          random_seed=1234, trace=db)

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

    az.plot_trace(trace, compact=True);
    plt.savefig('results/{}/trace_plot.png'.format(country))
    az.plot_posterior(trace);
    plt.savefig('results/{}/posterior_plot.png'.format(country))

    # ========== Compute predictions =============

    h = 7  # number points to prediction ahead
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
    dy_fit_final = np.percentile(trace['rate'], 50, axis=0) * y_fit_final * (1 - (y_fit_final / np.percentile(trace['K'], 50, axis=0)) ** np.percentile(trace['a'], 50, axis=0))

    # Plot prediction of comulative cases
    #ymax_limit = max(max(y_fit_final), df.acumulado.astype('float64').max()) * 1.10
    yref_ycoord_0 = min(np.median(y_fit_final), df.acumulado.median()) * 0.6
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plt.plot(df[x_data_label].values, df.acumulado.astype('float64').values,
             'b', marker='s', ls='-', lw=1, label='Observed data')
    plt.plot(np.arange(0, post_pred_final['y'].shape[1]), y_fit_final,
             'k', marker='^', ls=':', ms=5, mew=1, label='Model prediction')
    plt.fill_between(np.arange(0, post_pred_final['y'].shape[1]), y_min_final, y_max_final, color='0.5', alpha=0.5)
    ax.axvline(x_values[-1], ls='--', color='k', label='Day {} since {} cases'.format(x_values[-1],epidemic_start))
    plt.text(x_values[-1], yref_ycoord_0,
             "{}".format(df[df[x_data_label] == x_values[-1]].date.map(lambda x: str(x)[:10]).values[0]),
             {'color': 'k', 'fontsize': 14},
             horizontalalignment='right', verticalalignment='baseline', rotation=90, clip_on=False)
    plt.suptitle('Richard model prediction: {}'.format(country), fontsize=24)
    f_values = (round(trace['rate'].mean(), 2), round(trace['x0'].mean(), 2),
                round(trace['K'].mean(), 2), round(trace['R0'].mean(), 2))
    plt.title(
        'Growth rate: {}, Turning point: {}, \n Final epidemic size: {}, Basic reproduction number: {}'.format(
            *f_values), fontsize=14)
    ax.set(xlabel='Days since {} cases'.format(epidemic_start))
    ax.set(ylabel='Cumulative confirmed cases')
    plt.legend(loc='upper left')
    plt.savefig('results/{}/cumulative_prediction_plot.png'.format(country))

    # Incidence of new cases prediction
    yref_ycoord_0 = min(np.median(dy_fit_final), df.new_cases.median()) * 0.6
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plt.plot(df[x_data_label], df['new_cases'], 'b', marker='s', ls='-', label='Observed data')
    plt.plot(np.arange(0, post_pred_final['y'].shape[1]), dy_fit_final, 'k', marker='^', ls=':', ms=5, mew=1,
             label='Predicted incidence of new cases')
    ax.axvline(x_values[-1], ls='--', color='k', label='Day {} since {} cases'.format(x_values[-1],epidemic_start))
    plt.text(x_values[-1], yref_ycoord_0,
             "{}".format(df[df[x_data_label] == x_values[-1]].date.map(lambda x: str(x)[:10]).values[0]),
             {'color': 'k', 'fontsize': 14},
             horizontalalignment='right', verticalalignment='baseline', rotation=90, clip_on=False)
    plt.xlabel('Days since {} cases'.format(epidemic_start))
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
    df_pred[x_data_label] = np.arange(0, post_pred_final['y'].shape[1])
    df_pred['acumulado_pred'] = y_fit_final
    df_pred['acumulado_pred_lower'] = y_min_final
    df_pred['acumulado_pred_upper'] = y_max_final
    df_pred['new_cases_pred'] = dy_fit_final
    df_full = pd.merge(df_pred, df, on=x_data_label, how='left')
    df_full.index = df_pred.index
    df_full['date'] = df_pred.index
    cols_selected = ['date', x_data_label, 'new_cases', 'acumulado', 'acumulado_pred',
                     'acumulado_pred_lower', 'acumulado_pred_upper', 'new_cases_pred']
    df_full[cols_selected].to_csv('results/{}/{}_data_until_{}_predictions.csv'.format(country, country, dt_until),
                                  index=False, sep=',', decimal='.')
    df_full = df_full[cols_selected]
    print(df_full)

    # Interactive plots with plotly

    # Covid-19: Richards model prediction for Spain
    date_max_plot = str(df[df[x_data_label] == x_values[-1]].date.map(lambda x: str(x)[:10]).values[0])
    df = df_full.copy()
    ymax_limit = max(df.acumulado_pred.max(), df.acumulado.max()) * 1.10
    yref_ycoord = min(df.acumulado_pred.median(), df.acumulado.median()) * 0.6
    fig = go.Figure(layout={'title': 'Covid-19: Richards model prediction for {}'.format(country),
                            'xaxis_title': "Date (days)", 'yaxis_title': "Cumulative confirmed cases",
                            'font': {'family': 'Courier', 'size': 16}})
    fig.add_trace(go.Scatter(x=df.date, y=df.acumulado_pred,
                             fill=None, mode='markers+lines', line_color='red', name="Predicted"))
    fig.add_trace(go.Scatter(x=df.date, y=df.acumulado_pred_lower,
                             fill=None, mode='lines', line_color='gray', showlegend=False))
    fig.add_trace(go.Scatter(x=df.date, y=df.acumulado_pred_upper,
                             fill='tonexty',  # fill area between trace0 and trace1
                             mode='lines', line_color='gray', showlegend=False))
    fig.add_trace(go.Scatter(x=df.date, y=df.acumulado,
                             fill=None, mode='markers+lines', line_color='blue', name="Observed"))
    # Line Vertical
    fig.add_shape(dict(type="line", x0=date_max_plot, y0=0, x1=date_max_plot, y1=ymax_limit,
                       line=dict(color="gray", width=1)))
    fig.update_layout(annotations=[dict(x=date_max_plot, y=yref_ycoord, xref="x", text=date_max_plot, textangle=-90)])
    fig.update_layout(yaxis_tickformat='6.0f')
    fig.layout.template = 'plotly_white'
    py.plot(fig, filename='results/{}/comulative_prediction_plot_{}.html'.format(country, dt_until), validate=False)
    #fig.show()

    # Plot incidence of new cases
    ymax_limit = max(df.new_cases.max(), df.new_cases.max()) * 1.10
    yref_ycoord = min(df.new_cases.median(), df.new_cases.median()) * 0.6
    fig = go.Figure(layout={'title': 'Covid-19: Richards model prediction of new cases for {}'.format(country),
                            'font': {'family': 'Courier', 'size': 16}})
    fig.add_trace(go.Scatter(x=df.date, y=df.new_cases_pred,
                             fill=None, mode='markers+lines', line_color='red', name="Predicted"))
    fig.add_trace(go.Scatter(x=df.date, y=df.new_cases,
                             fill=None, mode='markers+lines', line_color='blue', name="Observed"))
    # Line Vertical
    fig.add_shape(dict(type="line", x0=date_max_plot, y0=0, x1=date_max_plot, y1=ymax_limit,
                       line=dict(color="gray", width=1)))
    fig.update_layout(annotations=[dict(x=date_max_plot, y=yref_ycoord, xref="x", text=date_max_plot, textangle=-90)])
    fig.update_layout(yaxis_tickformat='6.0f')
    fig.layout.template = 'plotly_white'
    py.plot(fig, filename='results/{}/newcases_prediction_plot_{}.html'.format(country, dt_until), validate=False)
