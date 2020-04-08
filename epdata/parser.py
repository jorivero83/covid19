import re
import pandas as pd


class EpData:

    def __init__(self):
        self.input_columns = ['anno', 'periodo', 'acumulado']
        self.final_columns = ['date', 'acumulado', 'new_cases']
        self.map_month_format = {'Enero': '01', 'Febrero': '02', 'Marzo': '03', 'Abril': '04',
                                 'Mayo': '05', 'Junio': '06', 'Julio': '07', 'Agosto': '08',
                                 'Septiembre': '09', 'Octubre': '10', 'Noviembre': '11',
                                 'Diciembre': '12'}

    def parser(self, df, ylabel='acumulado'):
        df_input = df.copy()
        df_input.columns = self.input_columns
        df_input = df_input[df_input[ylabel].notnull()]
        df_input['day'] = df_input.periodo.map(lambda x: str(re.findall(r'\w+', str(x))[1]).zfill(2))
        df_input['month'] = df_input.periodo.map(lambda x: re.findall(r'\w+', str(x))[-1]).map(self.map_month_format)
        df_input['date'] = pd.to_datetime(df_input['anno'] + '-' + df_input['month'] + '-' + df_input['day'])
        df_input = df_input.sort_values('date')
        df_input.index = df_input['date']
        df_input[ylabel] = df_input[ylabel].map(lambda x: str(x).replace('.', '')).astype(int)
        if 'new_cases' in self.final_columns:
            df_input['new_cases'] = df_input[ylabel] - df_input[ylabel].shift(1)

        return df_input[self.final_columns]
