import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from epdata.parser import EpData
from datetime import datetime, timedelta
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 100)

#K * (1 + np.exp(-r * a * (t - ti))) ** (-1 / a)
def richards_model(t, r, a, K, ti):
    return K * (1 + np.exp(-r * a * (t - ti - (np.log(a) / (r * a))))) ** (-1 / a)

if __name__ == '__main__':
    # Load data
    country = 'Spain'
    oper_file_path1 = 'data/spain_data.csv'
    df_input = pd.read_csv(oper_file_path1, sep=';', decimal=',', error_bad_lines=False)
    df_input = EpData().parser(df=df_input)

    # Prepare data for training the model
    df = df_input[df_input['acumulado'] > 100][['date', 'acumulado', 'new_cases']].copy()
    df['days_since_100'] = list(np.arange(df.shape[0]))
    print(df.tail())

    # Fit with curvet_fit
    xdata = [datetime.strptime(str(dt)[:10], '%Y-%m-%d') for dt in df.date.values]
    ydata = df.acumulado.astype('float64').values
    x_values = xdata[:36]
    y_values = ydata[:36]

    x_values_pred = x_values.copy()
    h = 20

    for i in range(h):
        x_values_pred.append(x_values_pred[-1] + timedelta(days=1))

    x = np.array(np.linspace(1, len(y_values), len(y_values), dtype=np.float))
    xp = np.array(np.linspace(x[0], x[-1] + h, len(x) + h), dtype=np.float)

    popt, pcov = curve_fit(richards_model, x, y_values, p0=[0.45, 1, 5000, 15])

    # Predictions: cumulative cases
    richards_infected_pred = richards_model(xp, *popt)
    err = np.sqrt(np.diag(pcov))
    richards_infected_upper = richards_model(xp, *(popt + err))
    richards_infected_lower = richards_model(xp, *(popt - err))

    # Predictions: incidence of new cases
    dydata = np.diff(ydata, n=1)
    dypred = popt[0] * richards_infected_pred * (1 - (richards_infected_pred / popt[2]) ** popt[1])
    # dypred_lower = popt[0]*richards_infected_lower*(1 - (richards_infected_lower/popt[2])**popt[1])
    # dypred_upper = popt[0]*richards_infected_upper*(1 - (richards_infected_upper/popt[2])**popt[1])

    # Plot the model prediction
    fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
    plt.suptitle('Richards model prediction: {} \n \n r=%5.2f, a=%5.2f, K=%5.2f, ti=%5.2f'.format(country) % tuple(popt), fontsize=24)
    axs[0].plot(x_values_pred, richards_infected_pred, 'ro-', label='Predicted')
    axs[0].plot(xdata, ydata, 'bp-', label='Observed')
    axs[0].fill_between(x_values_pred, richards_infected_lower, richards_infected_upper, alpha=.25, color='gray')
    axs[0].axvline(x_values[-1], ls='--', color='k',
                   label='Day {} since 100 cases (train/test split line)'.format(int(x[-1])))
    axs[0].text(x_values[-1], 10, "{}".format(str(x_values[-1])[:10]), {'color': 'k', 'fontsize': 10},
                horizontalalignment='right', verticalalignment='baseline', rotation=90, clip_on=False)
    axs[0].legend()
    axs[1].plot(xdata, np.append(0, dydata), 'gp-', label='Observed')
    axs[1].plot(x_values_pred, dypred, 'rp-', label='Predicted')
    axs[1].axvline(x_values[-1], ls='--', color='k',
                   label='Day {} since 100 cases (train/test split line)'.format(int(x[-1])))
    axs[1].text(x_values[-1], 10, "{}".format(str(x_values[-1])[:10]), {'color': 'k', 'fontsize': 10},
                horizontalalignment='right', verticalalignment='baseline', rotation=90, clip_on=False)
    tp_date = df[df.days_since_100 == int(round(popt[3]))].date.values[0]
    axs[1].axvline(tp_date, ls='-', color='k', label='Turning point: {}'.format(str(tp_date)[:10]))
    fig.autofmt_xdate(rotation=45)
    axs[1].legend()
    plt.savefig('results/{}/curve_fit_prediction_plot_{}.png'.format(country,str(str(x_values[-1])[:10]).replace('-','')))
    #plt.show()

    # Data of model prediction
    df_pred = pd.DataFrame(index=pd.date_range(start=df.index[0], freq='D', periods=len(richards_infected_pred)))
    df_pred['days_since_100'] = np.arange(0, len(richards_infected_pred))
    df_pred['acumulado_pred'] = richards_infected_pred
    # df_pred['acumulado_pred_lower'] = y_min_final
    # df_pred['acumulado_pred_upper'] = y_max_final
    df_pred['new_cases_pred'] = dypred
    df_full = pd.merge(df_pred, df, on='days_since_100', how='left')
    df_full.index = df_pred.index
    df_full['date'] = df_pred.index
    cols_selected = ['date', 'days_since_100', 'new_cases', 'acumulado', 'acumulado_pred',
                     'new_cases_pred']
    # df_full[cols_selected].to_csv('data/df_spain_20200401_with_predictions.csv', index=False, sep=',', decimal='.')
    print(df_full[cols_selected])


