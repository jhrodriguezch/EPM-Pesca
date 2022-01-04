"""
FUNCIONES DESAROLLADAS PARA EL MODELO DE PESCA DEL PROYECTO ENTRE LA PONTIFICIA UNIVERSIDAD JAVERIANA Y EMPRESAS
PÚBLICAS DE MEDELLÍN (PUJ - EPM)
"""
import numpy as np
import pandas as pd


class ModeloPesca:
    def __init__(self, wua, NeuData):
        # FIX WUA DATE
        wua_df = wua.copy()
        wua_df['Date'] = pd.to_datetime(wua_df['Date'], format='%d-%m-%Y')
        wua_df.sort_values(by=['Date'], inplace=True)
        wua_df.reset_index(inplace=True, drop=True)

        # SAVE EXTERNAL VARIABLES OBJECT
        self.wua_df = wua_df
        self.init_date = wua_df['Date'].iloc[0]
        self.end_date = wua_df['Date'].iloc[-1]

        # SAVE INTERNAL VARIABLES OBJECT
        self.dictionary = {'N. de pasos rk4': 10,
                           # Vulnerabilidad
                           'vulnerabilidad humedo': 0.216,
                           'vulnerabilidad seco': 0.216,
                           # Niveles de corte
                           'Nivel humedo (m)': 1.9,
                           'Nivel seco (m)': 1.0}

        # Pre calc values
        self.precalc_fun(NeuData)

        # PRINT ESCENTIAL DATA
        print('from {} to {}'.format(self.init_date, self.end_date))

    # MAIN FUNCTIONS
    def precalc_fun(self, NeuData) -> None:
        # Add neutro data
        self.wua_df['Mes'] = [str(ii) if ii > 9 else '0'+str(ii) for ii in pd.DatetimeIndex(self.wua_df['Date']).month]
        self.wua_df = pd.merge(self.wua_df, NeuData, how='left', on='Mes')

        # Add type month
        # 0 = Neutro
        # 1 = Nino
        # 2 = Nina
        self.wua_df['type'] = 0
        self.wua_df.loc[self.wua_df['Nivel'] <= self.dictionary['Nivel seco (m)'], 'type'] = 1
        self.wua_df.loc[self.wua_df['Nivel'] >= self.dictionary['Nivel humedo (m)'],'type'] = 2

        # Calc Vulnerability
        self.wua_df['Vulnerabilidad'] = 1.0
        self.wua_df.loc[self.wua_df['type'] == 1, 'Vulnerabilidad'] = self.dictionary['vulnerabilidad seco']
        self.wua_df.loc[self.wua_df['type'] == 2, 'Vulnerabilidad'] = self.dictionary['vulnerabilidad humedo']

        # Del
        self.wua_df.drop('Mes', axis = 1, inplace = True)

    def prereclutas_fun(self, param: dict):
        """
        Calc prereclutas time series
        :param param: Dictionary with parameters
        :return: Dataframe with results
        """
        df = pd.DataFrame()

        # ADD MAIN COLUMNS IN DATAFRAME
        df['Date'] = self.wua_df['Date']
        df['Mes'] = [str(ii) if ii >= 10 else '0' + str(ii) for ii in self.wua_df['Date'].dt.month]
        df = pd.merge(df, param['Porcentaje de abundancia'], on='Mes', how='left')
        df['WUA'] = self.wua_df['WUA']
        df['Area'] = self.wua_df['Area']

        # ADD DATAFRAMES CALCS

        #df['Vulnerabilidad - Reclutamiento'] = self.calc_vulnerabilidad()
        df['Vulnerabilidad - Fenomeno nina'] = self.wua_df['Vulnerabilidad']
        df.loc[self.wua_df['type'] != 2, 'Vulnerabilidad - Fenomeno nina'] = 1.0

        df['Vulnerabilidad - Fenomeno nino'] = self.wua_df['Vulnerabilidad']
        df.loc[self.wua_df['type'] != 1, 'Vulnerabilidad - Fenomeno nino'] = 1.0

        df['Individuos totales'] = self.wua_df['WUA'] * param['Individuos por ha']
        df['Individuos maduros'] = df['Individuos totales'] * param['Porcentaje hembras'] * param['Porcentaje maduros']
        df['Individuos reproductibles'] = df['Porcentaje'] * df['Individuos maduros']
        df['Huevos esperados'] = df['Individuos maduros'] * param['Desove por individuo']
        df['Pre-reclutas'] = \
            df['Individuos reproductibles'] * param['Porcentaje de supervivencia'] * param['Desove por individuo']

        # FIX DATAFRAME
        df.drop(['Mes'], axis=1, inplace=True)

        # SAVE
        self.pes_ha = param['Individuos por ha']
        return df

    def servicio_ecosistemico_fun(self, abundancia_init, pre_reclut_df: pd.DataFrame, param: dict):

        # VARIABLES
        reclutas_lst = []
        captura_lst = []
        muertes_desinund_lst = []
        muertes_lst = []
        abundancia_lst = [abundancia_init]

        pre_reclut_df['Peso por individuo'] = self.randomizer(param['Peso por individuo - promedio'],
                                                              param['Peso por individuo - des. estandar'],
                                                              param['Peso por individuo - mínimo'],
                                                              param['Peso por individuo - máximo'],
                                                              len(pre_reclut_df))

        pre_reclut_df['Muert. desinun'] = (self.wua_df['Area Neutro'] - pre_reclut_df['Area']) * self.pes_ha
        pre_reclut_df.loc[self.wua_df['type'] != 1, 'Muert. desinun'] = 0
        pre_reclut_df.loc[pre_reclut_df['Muert. desinun'] <= 0, 'Muert. desinun'] = 0

        # STOCK FUNCTION
        def dabundancia_dt(abundancia_ini, flow):
            """
            :param abundancia_ini: Abundancia inicial
            :param flow: list = [reclutas, tasa de mortalidad, capturas, muertes desinundacion, dt]
            :return: (reclutas - (abundancia + reclutas * dt) * Tasa mortalidad - muertes desinundacion)
            """
            died = (abundancia_ini + flow[0] * flow[4]) * flow[1] if ((abundancia_ini + flow[0] * flow[4]) *
                                                                      flow[1] > 0) else 0
            res = flow[0] - died - flow[2] - flow[3]
            return res

        for _, prerec_info in pre_reclut_df.iterrows():

            # FLOWS
            reclutas = prerec_info['Pre-reclutas'] * prerec_info['Vulnerabilidad - Fenomeno nina']

            muertes_desinund = prerec_info['Muert. desinun']
            muertes = (abundancia_init + reclutas * 1/self.dictionary['N. de pasos rk4']) * param['Tasa de mortalidad']\
                      + muertes_desinund if (abundancia_init + reclutas * 1/self.dictionary['N. de pasos rk4']) *\
                                            param['Tasa de mortalidad'] + muertes_desinund > 0 else 0

            captura = \
                param['Pescadores'] * param['Captura potencial promedio'] * \
                param['Porcentaje de captura'] * 30 / prerec_info['Peso por individuo']
            captura = captura if captura <= abundancia_init + reclutas - muertes - muertes_desinund else abundancia_init + reclutas - muertes - muertes_desinund

            # AUXILIAR
            cte = [reclutas, param['Tasa de mortalidad'], captura, 0, 1/self.dictionary['N. de pasos rk4']]

            # CALC
            for _ in np.arange(self.dictionary['N. de pasos rk4']):
                abundancia_tmp = rk4(dabundancia_dt, abundancia_init, cte, 1 / self.dictionary['N. de pasos rk4'])
                abundancia_init = abundancia_tmp

            # SAVE
            reclutas_lst.append(reclutas)
            captura_lst.append(captura)
            muertes_desinund_lst.append(muertes_desinund)
            muertes_lst.append(muertes)
            abundancia_lst.append(abundancia_init)

        abundancia_lst.pop()

        servicio_ecosistemico_res = pd.DataFrame()
        servicio_ecosistemico_res['Date'] = self.wua_df['Date']
        servicio_ecosistemico_res['Abundancia'] = abundancia_lst
        servicio_ecosistemico_res['Reclutas'] = reclutas_lst
        servicio_ecosistemico_res['Capturas'] = captura_lst
        servicio_ecosistemico_res['Muertes'] = muertes_lst
        servicio_ecosistemico_res['Muertes desinundación'] =muertes_desinund_lst

        self.pre_reclut_df = pre_reclut_df

        # DEL
        pre_reclut_df.drop('Muert. desinun', axis=1, inplace=True)
        return servicio_ecosistemico_res

    def capital_financiero_fun(self, cap_financiero_init: int, servicio_ecosistemico_res: pd.DataFrame(), param: dict):
        # VARIABLES
        ventas_lst = []
        costos_lst = []
        p_x_libra_lst = []

        cfinanciero_acum_lst = [cap_financiero_init]
        cfinanciero_lst = []
        servicio_ecosistemico_res['Peso por individuo'] = self.pre_reclut_df['Peso por individuo']

        # STOCK FUNCTION
        capital_financiero = lambda init, ctes: ctes[0] - ctes[1]

        for _, servicio_ecosistemico in servicio_ecosistemico_res.iterrows():
            # FLOWS
            p_x_libra = self.__rand_norm__(param['Precio por libra - promedio'],
                                           param['Precio por libra - minimo'],
                                           param['Precio por libra - maximo'],
                                           param['Precio por libra - desv. estandar'])

            ventas = servicio_ecosistemico['Capturas'] * (1 - param['Porcentaje de autoconsumo']) * \
                     servicio_ecosistemico['Peso por individuo'] * 2.2 * p_x_libra
            costos = param['Costos mensuales']

            # AUXILIAR
            cte = [ventas, costos]

            # CALC
            for _ in np.arange(self.dictionary['N. de pasos rk4']):
                cap_financiero_tmp = rk4(capital_financiero, cap_financiero_init,
                                         cte, 1 / self.dictionary['N. de pasos rk4'])
                cap_financiero_init = cap_financiero_tmp

            # SAVE
            p_x_libra_lst.append(p_x_libra)
            cfinanciero_acum_lst.append(cap_financiero_init)
            cfinanciero_lst.append(ventas - costos)
            ventas_lst.append(ventas)
            costos_lst.append(costos)

        cfinanciero_acum_lst.pop()

        capital_financiero_res = pd.DataFrame()
        capital_financiero_res['Date'] = servicio_ecosistemico_res['Date'].copy()
        capital_financiero_res['Capital financiero - acumulado'] = cfinanciero_acum_lst
        capital_financiero_res['Precio por libra'] = p_x_libra_lst
        capital_financiero_res['Capital financiero'] = cfinanciero_lst
        capital_financiero_res['Ventas'] = ventas_lst
        capital_financiero_res['Costos'] = costos_lst

        return capital_financiero_res

    # SECONDARY FUNCTIONS
    def randomizer(self, promedio: float, std: float, min_n: float, max_n: float, s: int):
        res = [self.__rand__(promedio, min_n, max_n, std) for _ in np.arange(s)]
        return res

    def __rand__(self, promedio: float, min_n: float, max_n: float, std: float):
        value = float(np.random.logistic(promedio, std, 1))
        if (value > min_n) & (value < max_n):
            return value
        else:
            return self.__rand__(promedio, min_n, max_n, std)

    def __rand_norm__(self, promedio: float, min_n: float, max_n: float, std: float):
        value = float(np.random.normal(promedio, std, 1))
        if (value > min_n) & (value < max_n):
            return value
        else:
            return self.__rand_norm__(promedio, min_n, max_n, std)


def rk4(fun, var_ini, cte, dt):
    """
    Runge Kutta 4th order
    :param fun: function
    :param var_ini: Any
    :param cte: List
    :param dt: Any
    :return: var_end: Any

    Example:
    fun = lambda var, cte: cte[0] - var * cte[1]
    fun(var) = cte[0] - var * cte[1]
    """
    k1 = dt * fun(var_ini, cte)
    k2 = dt * fun(var_ini + 1 / 2 * k1, cte)
    k3 = dt * fun(var_ini + 1 / 2 * k2, cte)
    k4 = dt * fun(var_ini + k3, cte)
    return var_ini + (k1 + 2 * k2 + 2 * k3 + k4) / 6.
