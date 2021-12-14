"""
FUNCIONES DESAROLLADAS PARA EL MODELO DE PESCA DEL PROYECTO ENTRE LA PONTIFICIA UNIVERSIDAD JAVERIANA Y EMPRESAS
PÚBLICAS DE MEDELLÍN (PUJ - EPM)
"""
import numpy as np
import pandas as pd


class ModeloPesca:
    def __init__(self, wua):
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
        self.dictionary = {'N. de pasos rk4': 10}

        # PRINT ESCENTIAL DATA
        print('from {} to {}'.format(self.init_date, self.end_date))

    # MAIN FUNCTIONS
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
        df['Vulnerabilidad - Reclutamiento'] = self.calc_vulnerabilidad()
        df['Individuos totales'] = self.wua_df['WUA'] * param['Individuos por ha']
        df['Individuos maduros'] = df['Individuos totales'] * param['Porcentaje hembras'] * param['Porcentaje maduros']
        df['Individuos reproductibles'] = df['Porcentaje'] * df['Individuos maduros']
        df['Huevos esperados'] = df['Individuos maduros'] * param['Desove por individuo']
        df['Pre-reclutas'] = \
            df['Individuos reproductibles'] * param['Porcentaje de supervivencia'] * param['Desove por individuo']

        # FIX DATAFRAME
        df.drop(['Mes'], axis=1, inplace=True)

        return df

    def servicio_ecosistemico_fun(self, abundancia_init, pre_reclut_df: pd.DataFrame, param: dict):

        # VARIABLES
        reclutas_lst = []
        captura_lst = []
        muertes_lst = []
        abundancia_lst = [abundancia_init]

        pre_reclut_df['Peso por individuo'] = self.randomizer(param['Peso por individuo - promedio'],
                                                              param['Peso por individuo - des. estandar'],
                                                              param['Peso por individuo - mínimo'],
                                                              param['Peso por individuo - máximo'],
                                                              len(pre_reclut_df))

        # STOCK FUNCTION
        def dabundancia_dt(abundancia_ini, flow):
            died = abundancia_ini * flow[1] if abundancia_ini * flow[1] > 0 else 0
            res = flow[0] - died - flow[2]
            return res

        # dabundancia_dt = lambda abundancia_ini, flow: flow[0] - abundancia_ini * flow[1] - flow[2]

        for _, prerec_info in pre_reclut_df.iterrows():

            # FLOWS
            reclutas = prerec_info['Pre-reclutas'] * prerec_info['Vulnerabilidad - Reclutamiento']
            captura = \
                param['Pescadores'] * param['Captura potencial promedio'] * \
                param['Porcentaje de captura'] * 30 / prerec_info['Peso por individuo']
            muertes = abundancia_init * param['Tasa de mortalidad'] if abundancia_init * param['Tasa de mortalidad'] > 0 else 0

            # AUXILIAR
            cte = [reclutas, param['Tasa de mortalidad'], captura]

            # CALC
            for _ in np.arange(self.dictionary['N. de pasos rk4']):
                abundancia_tmp = rk4(dabundancia_dt, abundancia_init, cte, 1 / self.dictionary['N. de pasos rk4'])
                abundancia_init = abundancia_tmp

            # SAVE
            reclutas_lst.append(reclutas)
            captura_lst.append(captura)
            muertes_lst.append(muertes)
            abundancia_lst.append(abundancia_init)

        abundancia_lst.pop()

        servicio_ecosistemico_res = pd.DataFrame()
        servicio_ecosistemico_res['Date'] = self.wua_df['Date']
        servicio_ecosistemico_res['Abundancia'] = abundancia_lst
        servicio_ecosistemico_res['Reclutas'] = reclutas_lst
        servicio_ecosistemico_res['Capturas'] = captura_lst
        servicio_ecosistemico_res['Muertes'] = muertes_lst

        self.pre_reclut_df = pre_reclut_df

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

    def calc_vulnerabilidad(self):
        # Vulnerabilidad
        vul_humedo: float = 0.216
        vul_seco: float = 1.1

        # Niveles de corte
        nivel_humedo: float = 1.9
        nivel_seco: float = 1.0

        # main
        vulnerability = []
        for nivel in self.wua_df['Nivel'].to_list():
            if nivel >= nivel_humedo:
                vulnerability.append(vul_humedo)
            elif nivel <= nivel_seco:
                vulnerability.append(vul_seco)
            else:
                vulnerability.append(1.0)
        return vulnerability


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
