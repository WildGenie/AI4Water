
import glob
import warnings
from typing import Union, Tuple

try:
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
except (ModuleNotFoundError, OSError):
    shape, mapping, unary_union = None, None, None

from ai4water.backend import os, shapefile, xr, np, pd, fiona

from .utils import check_attributes
from .datasets import Datasets, _maybe_download

from ai4water.preprocessing.resample import Resampler
from ai4water.preprocessing.spatial_utils import find_records


class MtropicsLaos(Datasets):
    """
    Downloads and prepares hydrological, climate and land use data for Laos from
    Mtropics_ website and ird_ data servers.

    Methods
    -------
        - fetch_lu
        - fetch_ecoli
        - fetch_rain_gauges
        - fetch_weather_station_data
        - fetch_pcp
        - fetch_hydro
        - make_regression

    .. _Mtropics:
        https://mtropics.obs-mip.fr/catalogue-m-tropics/

    .. _ird:
        https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/EWOYNK
    """
    target = ['Ecoli_mpn100']

    url = {
        'lu.zip':
"https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=0f1aea48-2a51-9b42-7688-a774a8f75e7a",
        'pcp.zip':
"https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=3c870a03-324b-140d-7d98-d3585a63e6ec",
        'hydro.zip':
"https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=389bbea0-7279-12c1-63d0-cfc4a77ded87",
        'rain_guage.zip':
"https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=7bc45591-5b9f-a13d-90dc-f2a75b0a15cc",
        'weather_station.zip':
"https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=353d7f00-8d6a-2a34-c0a2-5903c64e800b",
        'ecoli_data.csv':
"https://dataverse.ird.fr/api/access/datafile/5435",
        "ecoli_dict.csv":
"https://dataverse.ird.fr/api/access/datafile/5436",
        "soilmap.zip":
"https://dataverse.ird.fr/api/access/datafile/5430",
        "subs1.zip":
"https://dataverse.ird.fr/api/access/datafile/5432",
        "suro.zip":
"https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=f06cb605-7e59-4ba4-8faf-1beee35d2162",
        "surf_feat.zip":
"https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=72d9e532-8910-48d2-b9a2-6c8b0241825b",
    }

    physio_chem_features = {
        "T_deg": "T",
        "EC_s/cm": "EC",
        "DO_percent": "DOpercent",
        "DO_mgl": "DO",
        "pH": "pH",
        "ORP_mV": "ORP",  # stream water oxidation-reduction potential
        "Turbidity_NTU": "Turbidity",
        "TSS_gL": "TSS",

                            }

    weather_station_data = ['air_temp', 'rel_hum', 'wind_speed', 'sol_rad']
    inputs = weather_station_data + ['water_level', 'pcp', 'susp_pm']

    def __init__(self, **kwargs):

        if xr is None:
            raise ModuleNotFoundError("xarray must be installed to use datasets sub-module")

        super().__init__(**kwargs)
        self._download()

        # we need to pre-process the land use shapefiles
        in_dir = os.path.join(self.ds_dir, 'lu')
        out_dir = os.path.join(self.ds_dir, 'lu1')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        files = glob.glob(f'{in_dir}/*.shp')
        for fpath in files:
            f = os.path.basename(fpath)
            shp_file = os.path.join(in_dir, f)
            op = os.path.join(out_dir, f)

            _process_laos_shpfiles(shp_file, op)

    def surface_features(
            self,
            st: Union[str, int, pd.Timestamp] = '2000-10-14',
            en: Union[str, int, pd.Timestamp] = '2016-11-12',
    )->pd.DataFrame:
        """soil surface features data"""
        fname = os.path.join(self.ds_dir, "surf_feat", "SEDOO_EdS_Houay Pano.xlsx")
        df = pd.read_excel(fname, sheet_name="Soil surface features")

        df.index = pd.to_datetime(df.pop('Date'))

        if st:
            if isinstance(en, int):
                assert isinstance(en, int)
                df = df.iloc[st:en]
            else:
                df = df.loc[st:en]
        return df

    def fetch_suro(
            self,
    )->pd.DataFrame:
        """returns surface runoff and soil detachment data from Houay pano, Laos PDR.

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (293, 13)
        Examples
        --------
            >>> from ai4water.datasets import MtropicsLaos
            >>> laos = MtropicsLaos()
            >>> suro = laos.fetch_suro()
        """
        fname = os.path.join(self.ds_dir, 'suro', 'SEDOO_Runoff_Detachment_Houay Pano.xlsx')
        df = pd.read_excel(fname, sheet_name="Surface runoff soil detachment")

        return df.dropna()

    def fetch_lu(self, processed=False):
        """returns landuse_ data as list of shapefiles.

        .. _landuse:
            https://doi.org/10.1038/s41598-017-04385-2"""
        lu_dir = os.path.join(self.ds_dir, f"{'lu1' if processed else 'lu'}")
        files = glob.glob(f'{lu_dir}/*.shp')
        return files

    def fetch_physiochem(
            self,
            features: Union[list, str] = 'all',
            st: Union[str, pd.Timestamp] = '20110525 10:00:00',
            en: Union[str, pd.Timestamp] = '20210406 15:05:00',
    ) -> pd.DataFrame:
        """
        Fetches physio-chemical features of Huoy Pano catchment Laos.

        Parameters
        ----------
            st :
                start of data.
            en :
                end of data.
            features :
                The physio-chemical features to fetch. Following features
                are available

                - 'T',
                - 'EC',
                - 'DOpercent',
                - 'DO',
                - 'pH',
                - 'ORP',
                - 'Turbidity',
                - 'TSS'

        Returns
        -------
            a pandas dataframe
        """

        if isinstance(features, list):
            _features = []
            for f in features:
                _features.append(self.physio_chem_features[f])
        else:
            assert isinstance(features, str)
            if features == 'all':
                _features = features
            else:
                _features = self.physio_chem_features[features]

        features = check_attributes(_features, list(self.physio_chem_features.values()))

        fname = os.path.join(self.ds_dir, 'ecoli_data.csv')
        df = pd.read_csv(fname, sep='\t')
        df.index = pd.to_datetime(df['Date_Time'])

        df = df[features]

        col_names = {v: k for k, v in self.physio_chem_features.items() if v in features}

        df = df.rename(columns=col_names)

        return df.loc[st:en]

    def fetch_ecoli(
            self,
            features: Union[list, str] = 'Ecoli_mpn100',
            st: Union[str, pd.Timestamp] = '20110525 10:00:00',
            en: Union[str, pd.Timestamp] = '20210406 15:05:00',
            remove_duplicates: bool = True,
    ) -> pd.DataFrame:
        """
        Fetches E. coli data collected at the outlet. See Ribolzi_ et al., 2021
        and Boithias_ et al., 2021 for reference.
        NaNs represent missing values. The data is randomly sampled between 2011
        to 2021 during rainfall events. Total 368 E. coli observation points are available now.

        Parameters
        ----------
            st :
                start of data. By default the data is fetched from the point it
                is available.
            en :
                end of data. By default the data is fetched til the point it is
                available.
            features :
                E. coli concentration data. Following data are available

                - Ecoli_LL_mpn100: Lower limit of the confidence interval
                - Ecoli_mpn100: Stream water Escherichia coli concentration
                - Ecoli_UL_mpn100: Upper limit of the confidence interval
            remove_duplicates :
                whether to remove duplicates or not. This is because
                some values were recorded within a minute,

        Returns
        -------
            a pandas dataframe consisting of features as columns.

        .. _Ribolzi:
            https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/EWOYNK

        .. _Boithias:
            https://doi.org/10.1002/hyp.14126

        """
        fname = os.path.join(self.ds_dir, 'ecoli_data.csv')
        df = pd.read_csv(fname, sep='\t')
        df.index = pd.to_datetime(df['Date_Time'])

        available_features = {
            "Ecoli_LL_mpn100": "E-coli_4dilutions_95%-CI-LL",  # Lower limit of the confidence interval
            "Ecoli_mpn100": "E-coli_4dilutions",  # Stream water Escherichia coli concentration
            "Ecoli_UL_mpn100": "E-coli_4dilutions_95%-CI-UL"  # Upper limit of the confidence interval
        }
        if isinstance(features, list):
            _features = []
            for f in features:
                _features.append(available_features[f])
        else:
            assert isinstance(features, str)
            if features == 'all':
                _features = features
            else:
                _features = available_features[features]

        features = check_attributes(_features, list(available_features.values()))

        if remove_duplicates:
            df = df[~df.index.duplicated(keep='first')]

        df = df.sort_index()

        df = df[features]

        col_names = {v: k for k, v in available_features.items() if v in features}

        df = df.rename(columns=col_names)

        return df.loc[st:en]

    def fetch_rain_gauges(
            self,
            st: Union[str, pd.Timestamp] = "20010101",
            en: Union[str, pd.Timestamp] = "20191231",
    ) -> pd.DataFrame:
        """
        fetches data from 7 rain gauges_ which is collected at daily time step
        from 2001 to 2019.

        Parameters
        ----------
            st :
                start of data. By default the data is fetched from the point it
                is available.
            en :
                end of data. By default the data is fetched til the point it is
                available.
        Returns
        -------
            a dataframe of 7 columns, where each column represnets a rain guage
            observations. The length of dataframe depends upon range defined by
            `st` and `en` arguments.

        Examples
        --------
            >>> from ai4water.datasets import MtropicsLaos
            >>> laos = MtropicsLaos()
            >>> rg = laos.fetch_rain_gauges()

        .. _gauges:
            https://doi.org/10.1038/s41598-017-04385-2
        """
        # todo, does nan means 0 rainfall?
        fname = os.path.join(self.ds_dir, 'rain_guage', 'rain_guage.nc')
        if not os.path.exists(fname):
            files = glob.glob(f"{os.path.join(self.ds_dir, 'rain_guage')}/*.xlsx")
            df = pd.DataFrame()
            for f in files:
                _df = pd.read_excel(f, sheet_name='Daily',
                                    usecols=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'],
                                    keep_default_na=False)
                df = pd.concat([df, _df])

            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            df = df.reset_index(drop=True)  # index is of type Int64Index
            df.to_xarray().to_netcdf(fname)

        else:  # feather file already exists so load from it
            df = xr.load_dataset(fname).to_dataframe()

        df.index = pd.date_range('20010101', periods=len(df), freq='D')

        return df[st:en]

    def fetch_weather_station_data(
            self,
            st: Union[str, pd.Timestamp] = "20010101 01:00:00",
            en: Union[str, pd.Timestamp] = "20200101 00:00:00",
            freq: str = 'H'
    ) -> pd.DataFrame:
        """
        fetches hourly weather [1]_ station data which consits of air temperature,
        humidity, wind speed and solar radiation.

        Parameters
        ----------
            st :
                start of data to be feteched.
            en :
                end of data to be fetched.
            freq :
                frequency at which the data is to be fetched.
        Returns
        -------
            a pandas dataframe consisting of 4 columns

        .. [1]:
            https://doi.org/10.1038/s41598-017-04385-2
        """

        fname = os.path.join(self.ds_dir, 'weather_station', 'weather_stations.nc')
        if not os.path.exists(fname):
            files = glob.glob(f"{os.path.join(self.ds_dir, 'weather_station')}/*.xlsx")
            df = pd.DataFrame()
            for f in files:
                _df = pd.read_excel(f, sheet_name='Hourly',
                                    usecols=['T', 'H', 'W', 'Gr'], keep_default_na=False)
                df = pd.concat([df, _df])

            # non-numertic dtype causes problem in converting/saving netcdf
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            df = df.reset_index(drop=True)  # index is of type Int64Index
            df.to_xarray().to_netcdf(fname)
        else:  # feather file already exists so load from it
            df = xr.load_dataset(fname).to_dataframe()

        df.columns = self.weather_station_data

        df.index = pd.date_range('20010101 01:00:00', periods=len(df), freq='H')

        return df[st:en]

    def fetch_pcp(self,
                  st: Union[str, pd.Timestamp] = '20010101 00:06:00',
                  en: Union[str, pd.Timestamp] = '20200101 00:06:00',
                  freq: str = '6min'
                  ) -> pd.DataFrame:
        """
        Fetches the precipitation_ data which is collected at 6 minutes time-step
        from 2001 to 2020.

        Parameters
        ----------
            st :
                starting point of data to be fetched.
            en :
                end point of data to be fetched.
            freq :
                frequency at which the data is to be returned.

        Returns
        -------
            pandas dataframe of precipitation data

        .. _precipitation:
            https://doi.org/10.1038/s41598-017-04385-2
        """
        # todo allow change in frequency

        fname = os.path.join(self.ds_dir, 'pcp', 'pcp.nc')
        # feather file does not exist
        if not os.path.exists(fname):
            files = glob.glob(f"{os.path.join(self.ds_dir, 'pcp')}/*.xlsx")
            df = pd.DataFrame()
            for f in files:
                _df = pd.read_excel(f, sheet_name='6mn', usecols=['Rfa'])
                df = pd.concat([df, _df])

            df = df.reset_index(drop=True)
            df.to_xarray().to_netcdf(fname)
        else:  # feather file already exists so load from it
            df = xr.load_dataset(fname).to_dataframe()

        df.index = pd.date_range('20010101 00:06:00', periods=len(df), freq='6min')
        df.columns = ['pcp']

        return df[st:en]

    def fetch_hydro(
            self,
            st: Union[str, pd.Timestamp] = '20010101 00:06:00',
            en: Union[str, pd.Timestamp] = '20200101 00:06:00',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        fetches water level (cm) and suspended particulate matter (g L-1). Both
        data are from 2001 to 2019 but are randomly sampled.

        Parameters
        ----------
            st : optional
                starting point of data to be fetched.
            en : optional
                end point of data to be fetched.
        Returns
        -------
            a tuple of pandas dataframes of water level and suspended particulate
            matter.
        """
        wl_fname = os.path.join(self.ds_dir, 'hydro', 'wl.nc')
        spm_fname = os.path.join(self.ds_dir, 'hydro', 'spm.nc')
        if not os.path.exists(wl_fname):
            print("reading data from xlsx files and saving them in netcdf format.")
            print("This will happen only once but will save io time.")
            files = glob.glob(f"{os.path.join(self.ds_dir, 'hydro')}/*.xlsx")
            wl = pd.DataFrame()
            spm = pd.DataFrame()
            for f in files:
                _df = pd.read_excel(f, sheet_name='Aperiodic')
                _wl = _df[['Date', 'Time', 'RWL04']]
                _wl.index = pd.to_datetime(_wl['Date'].astype(str) + ' ' + _wl['Time'].astype(str))
                _spm = _df[['Date.1', 'Time.1', 'SPM04']]
                _spm = _spm.iloc[_spm.first_valid_index():_spm.last_valid_index()]
                if os.path.basename(f) == 'OMPrawdataLaos2016.xlsx':
                    _spm.iloc[166] = ['2016-07-01', '20:43:47', 1.69388]
                    _spm.iloc[247] = ['2016-07-23', '12:57:47', 8.15714]
                    _spm.iloc[248] = ['2016-07-23', '17:56:47', 0.5]
                    _spm.iloc[352] = ['2016-08-16', '03:08:17', 1.12711864406]
                if os.path.basename(f) == 'OMPrawdataLaos2017.xlsx':
                    _spm.index = pd.to_datetime(_spm['Date.1'].astype(str))
                else:
                    _spm.index = pd.to_datetime(_spm['Date.1'].astype(str) + ' ' + _spm['Time.1'].astype(str))
                wl = pd.concat([wl, _wl['RWL04']])
                spm = pd.concat([spm, _spm['SPM04']])

            wl.columns = ['water_level']
            wl = wl.reset_index()
            wl.to_xarray().to_netcdf(wl_fname)
            spm.columns = ['susp_pm']
            spm = spm.reset_index()
            spm.to_xarray().to_netcdf(spm_fname)
        else:
            wl = xr.load_dataset(wl_fname).to_dataframe(['index'])
            spm = xr.load_dataset(spm_fname).to_dataframe(['index'])

        # wl.index = pd.to_datetime(wl.pop('index'))
        # spm.index = pd.to_datetime(spm.pop('index'))
        wl = wl[~wl.index.duplicated(keep='first')]
        spm = spm[~spm.index.duplicated(keep='first')]

        # FutureWarning: Value based partial slicing on non-monotonic DatetimeIndexes
        return wl.loc[st:en], spm.loc[st:en]

    def make_classification(
            self,
            input_features: Union[None, list] = None,
            output_features: Union[str, list] = None,
            st: Union[None, str] = "20110525 14:00:00",
            en: Union[None, str] = "20181027 00:00:00",
            freq: str = "6min",
            threshold: Union[int, dict] = 400,
            lookback_steps: int = None,
    ) -> pd.DataFrame:
        """
        Makes a classification problem.

        Parameters
        ----------
            input_features :
                names of inputs to use.
            output_features :
                feature/features to consdier as target/output/label
            st :
                starting date of data
            en :
                end date of data
            freq :
                frequency of data
            threshold :
                threshold to use to determine classes. Values greater than
                equal to threshold are set to 1 while values smaller than threshold
                are set to 0. The value of 400 is chosen for E. coli to make the
                the number 0s and 1s balanced. It should be noted that US-EPA recommends
                threshold value of 400 cfu/ml.
            lookback_steps:
                the number of previous steps to use. If this argument is used,
                the resultant dataframe will have (ecoli_observations * lookback_steps)
                rows. The resulting index will not be continuous.

        Returns
        -------
            a dataframe of shape `(inputs+target, st:en)`

        Example:
            >>> from ai4water.datasets import MtropicsLaos
            >>> laos = MtropicsLaos()
            >>> df = laos.make_classification()
        """
        thresholds = {
            'Ecoli_mpn100': 400
        }

        target: list = check_attributes(output_features, self.target)

        data = self._make_ml_problem(input_features, target, st, en, freq)

        if len(target) == 1:
            threshold = threshold or thresholds[target[0]]
        else:
            raise ValueError

        s = data[target[0]]
        s[s < threshold] = 0
        s[s >= threshold] = 1

        data[target[0]] = s

        if lookback_steps:
            return consider_lookback(data, lookback_steps, target)
        return data

    def make_regression(
            self,
            input_features: Union[None, list] = None,
            output_features: Union[str, list] = "Ecoli_mpn100",
            st: Union[None, str] = "20110525 14:00:00",
            en: Union[None, str] = "20181027 00:00:00",
            freq: str = "6min",
            lookback_steps: int = None
    ) -> pd.DataFrame:
        """
        Makes a regression problem using hydrological, environmental,
        and water quality data of Huoay pano.

        Parameters
        ----------
            input_features :
                names of inputs to use. By default following features
                are used as input

                - ``air_temp``
                - ``rel_hum``
                - ``wind_speed``
                - ``sol_rad``
                - ``water_level``
                - ``pcp``
                - ``susp_pm``

            output_features : feature/features to consdier as target/output/label
            st :
                starting date of data
            en :
                end date of data
            freq : frequency of data
            lookback_steps:
                the number of previous steps to use. If this argument is used,
                the resultant dataframe will have (ecoli_observations * lookback_steps)
                rows. The resulting index will not be continuous.

        Returns
        -------
            a dataframe of shape (inputs+target, st - en)

        Example
        -------
            >>> from ai4water.datasets import MtropicsLaos
            >>> laos = MtropicsLaos()
            >>> ins = ['pcp', 'air_temp']
            >>> out = ['Ecoli_mpn100']
            >>> reg_data = laos.make_regression(ins, out, '20110101', '20181231')

        todo add HRU definition
        """
        data = self._make_ml_problem(input_features, output_features, st, en, freq)

        if lookback_steps:
            return consider_lookback(data, lookback_steps, output_features)
        return data

    def _make_ml_problem(self, input_features, output_features, st, en, freq):
        inputs = check_attributes(input_features, self.inputs)
        target = check_attributes(output_features, self.target)
        features_to_fetch = inputs + target

        pcp = self.fetch_pcp(st=st, en=en)
        pcp = pcp.interpolate('linear', limit=5)
        pcp = pcp.fillna(0.0)

        w = self.fetch_weather_station_data(st=st, en=en)
        w = w.interpolate()
        w = w.bfill()
        assert int(w.isna().sum().sum()) == 0

        w.columns = ['air_temp', 'rel_hum', 'wind_speed', 'sol_rad']
        w_6min = Resampler(w,
                           freq=freq,
                           how={'air_temp': 'linear',
                                'rel_hum': 'linear',
                                'wind_speed': 'linear',
                                'sol_rad': 'linear'
                                }
                           )()

        ecoli = self.fetch_ecoli(st=st, en=en)
        ecoli = ecoli.dropna()
        ecoli_6min = ecoli.resample(freq).mean()

        wl, spm = self.fetch_hydro(st=st, en=en)
        wl_6min = wl.resample(freq).first().interpolate(method="linear")
        spm_6min = spm.resample(freq).first().interpolate(method='linear')

        data = pd.concat([w_6min.loc[st:en],
                          pcp.loc[st:en],
                          wl_6min.loc[st:en],
                          spm_6min.loc[st:en],
                          ecoli_6min.loc[st:en],
                          ], axis=1)

        if data['water_level'].isna().sum() < 15:
            data['water_level'] = data['water_level'].bfill()  # only 11 nan present at start
            data['water_level'] = data['water_level'].ffill()  # only 1 nan is present at ned

        if data['susp_pm'].isna().sum() < 40:
            data['susp_pm'] = data['susp_pm'].bfill()  # only 26 nan is present at ned
            data['susp_pm'] = data['susp_pm'].ffill()  # only 9 nan is present

        return data.loc[st:en, features_to_fetch]


class MtropcsThailand(Datasets):
    url = {
        "pcp.zip":
            "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=27c65b5f-59cb-87c1-4fdf-628e6143d8c4",
        # "hydro.zip":
        #    "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=9e6f7144-8984-23bd-741a-06378fabd72",
        "rain_gauge.zip":
            "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=0a12ffcf-42bc-0289-1c55-a769ef19bb16",
        "weather_station.zip":
            "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=fa0bca5f-caee-5c68-fed7-544fe121dcf5 "
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._download()


class MtropicsVietnam(Datasets):
    url = {
        "pcp.zip":
            "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=d74ab1b0-379b-71cc-443b-662a73b7f596",
        "hydro.zip":
            "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=85fb6717-4095-a2a2-34b5-4f1b70cfd304",
        # "lu.zip":
        #    "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=c3724992-a043-4bbf-8ac1-bc6f9a608c1c",
        "rain_guage.zip":
            "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=3d3382d5-08c1-2595-190b-8568a1d2d6af",
        "weather_station.zip":
            "https://services.sedoo.fr/mtropics/data/v1_0/download?collectionId=8df40086-4232-d8d0-a1ed-56c860818989"
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._download()


def _process_laos_shpfiles(shape_file, out_path):

    if fiona is None:
        warnings.warn("preprocessing of shapefiles can not be done because no fiona installation is found.")
        return

    shp_reader = shapefile.Reader(shape_file)

    container = {
        'Forest': [],
        'Culture': [],
        'Fallow': [],
        'Teak': [],
        # 'others': []
    }

    for i in range(shp_reader.numRecords):
        lu = find_records(shape_file, 'LU3', i)
        shp = shp_reader.shape(i)
        if shp.shapeType == 0:
            continue
        geom = shape(shp.__geo_interface__)
        if lu.startswith('Forest'):
            container['Forest'].append(geom)
        elif lu.startswith('Culture'):
            container['Culture'].append(geom)
        elif lu.startswith('Fallow'):
            container['Fallow'].append(geom)
        elif lu.startswith('Teak'):
            container['Teak'].append(geom)
        else:  # just consider all others as 'culture' for siplicity
            container['Culture'].append(geom)
            # container['others'].append(geom)

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon' if os.path.basename(shape_file) in ['LU2000.shp', 'LU2001.shp'] else 'MultiPolygon',
        'properties': {'id': 'int',
                       'NAME': 'str',
                       'area': 'float'},
    }

    # Write a new Shapefile
    with fiona.open(out_path, 'w', 'ESRI Shapefile', schema) as c:
        for idx, lu in enumerate(list(container.keys())):
            geoms = container[lu]
            poly = unary_union([shape(s.__geo_interface__) for s in geoms])

            assert poly.is_valid

            c.write({
                'geometry': mapping(poly),
                'properties': {'id': idx,
                               'NAME': lu,
                               'area': poly.area},
            })


def consider_lookback(df:pd.DataFrame, lookback:int, col_name:str)->pd.DataFrame:
    """selects rows from dataframe considering lookback based upon nan
     values in col_name"""

    if isinstance(col_name, list):
        assert len(col_name) == 1
        col_name = col_name[0]

    if not isinstance(col_name, str):
        raise NotImplementedError

    start = False
    steps = 0

    masks = np.full(len(df), False)

    for idx, ecoli in enumerate(df[col_name].values[::-1]):
        if not ecoli != ecoli:
            start = True
            steps = 0

        if start and steps < lookback:
            masks[idx] = True
            steps += 1

        # if we have started counting but the limit has reached
        if start and steps > lookback:
            start = False

    return df.iloc[masks[::-1]]


def ecoli_mekong(
        st: Union[str, pd.Timestamp, int] = "20110101",
        en: Union[str, pd.Timestamp, int] = "20211231",
        features:Union[str, list] = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river (Houay Pano) area from 2011 to 2021 [1]_.

    Parameters
    ----------
        st : optional
            starting time. The default starting point is 2011-05-25 10:00:00
        en : optional
            end time, The default end point is 2021-05-25 15:41:00
        features : str, optional
            names of features to use. use ``all`` to get all features. By default
            following input features are selected

                ``station_name`` name of station/catchment where the observation was made
                ``T`` temperature
                ``EC`` electrical conductance
                ``DOpercent`` dissolved oxygen concentration
                ``DO`` dissolved oxygen saturation
                ``pH`` pH
                ``ORP`` oxidation-reduction potential
                ``Turbidity`` turbidity
                ``TSS`` total suspended sediment concentration
                ``E-coli_4dilutions`` Eschrechia coli concentration

        overwrite : bool
            whether to overwrite the downloaded file or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (1602, 10)

    Examples
    --------
        >>> from ai4water.datasets import ecoli_mekong
        >>> ecoli = ecoli_mekong()

    .. [1]
        https://essd.copernicus.org/preprints/essd-2021-440/
    """
    ecoli = ecoli_houay_pano(st, en, features, overwrite=overwrite)
    ecoli1 = ecoli_mekong_2016(st, en, features, overwrite=overwrite)
    ecoli2 = ecoli_mekong_laos(st, en, features, overwrite=overwrite)
    return pd.concat([ecoli, ecoli1, ecoli2])


def ecoli_mekong_2016(
        st: Union[str, pd.Timestamp, int] = "20160101",
        en: Union[str, pd.Timestamp, int] = "20161231",
        features:Union[str, list] = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river from 2016 from 29 catchments

    Parameters
    ----------
        st :
            starting time
        en :
            end time
        features : str, optional
            names of features to use. use ``all`` to get all features.
        overwrite : bool
            whether to overwrite the downloaded file or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (58, 10)

    Examples
    --------
        >>> from ai4water.datasets import ecoli_mekong_2016
        >>> ecoli = ecoli_mekong_2016()

    .. url_
        https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/ZRSBM4
    """
    url = {"ecoli_mekong_2016.csv": "https://dataverse.ird.fr/api/access/datafile/8852"}

    ds_dir = os.path.join(os.path.dirname(__file__), 'data', 'ecoli_mekong_2016')

    return _fetch_ecoli(ds_dir, overwrite, url, None, features, st, en,
                        "ecoli_houay_pano_tab_file")


def ecoli_houay_pano(
        st: Union[str, pd.Timestamp, int] = "20110101",
        en: Union[str, pd.Timestamp, int] = "20211231",
        features:Union[str, list] = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river (Houay Pano) area.

    Parameters
    ----------
        st : optional
            starting time. The default starting point is 2011-05-25 10:00:00
        en : optional
            end time, The default end point is 2021-05-25 15:41:00
        features : str, optional
            names of features to use. use ``all`` to get all features. By default
            following input features are selected

                ``station_name`` name of station/catchment where the observation was made
                ``T`` temperature
                ``EC`` electrical conductance
                ``DOpercent`` dissolved oxygen concentration
                ``DO`` dissolved oxygen saturation
                ``pH`` pH
                ``ORP`` oxidation-reduction potential
                ``Turbidity`` turbidity
                ``TSS`` total suspended sediment concentration
                ``E-coli_4dilutions`` Eschrechia coli concentration

        overwrite : bool
            whether to overwrite the downloaded file or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (413, 10)

    Examples
    --------
        >>> from ai4water.datasets import ecoli_houay_pano
        >>> ecoli = ecoli_houay_pano()

    .. url_
        https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/EWOYNK
    """
    url = {"ecoli_houay_pano_file.csv": "https://dataverse.ird.fr/api/access/datafile/9230"}

    ds_dir = os.path.join(os.path.dirname(__file__), 'data', 'ecoli_houay_pano')

    return _fetch_ecoli(ds_dir, overwrite, url, None, features, st, en,
                        "ecoli_houay_pano_tab_file")


def ecoli_mekong_laos(
        st: Union[str, pd.Timestamp, int] = "20110101",
        en: Union[str, pd.Timestamp, int] = "20211231",
        features:Union[str, list] = None,
        station_name:str = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river (Northern Laos).

    Parameters
    ----------
        st :
            starting time
        en :
            end time
        station_name : str
        features : str, optional
        overwrite : bool
            whether to overwrite or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (1131, 10)

    Examples
    --------
        >>> from ai4water.datasets import ecoli_mekong_laos
        >>> ecoli = ecoli_mekong_laos()

    .. url_
        https://dataverse.ird.fr/file.xhtml?fileId=9229&version=3.0
    """
    url = {"ecoli_mekong_loas_file.csv": "https://dataverse.ird.fr/api/access/datafile/9229"}

    ds_dir = os.path.join(os.path.dirname(__file__), 'data', 'ecoli_mekong_loas')

    return _fetch_ecoli(ds_dir, overwrite, url, station_name, features, st, en,
                        "ecoli_mekong_laos_tab_file")


def _fetch_ecoli(ds_dir, overwrite, url, station_name, features, st, en, _name):

    _maybe_download(ds_dir, overwrite, url, _name=_name)
    all_files = os.listdir(ds_dir)
    assert len(all_files)==1
    fname = os.path.join(ds_dir, all_files[0])
    df = pd.read_csv(fname, sep='\t')

    df.index = pd.to_datetime(df['Date_Time'])

    if station_name is not None:
        assert station_name in df['River'].unique().tolist()
        df = df.loc[df['River']==station_name]

    if features is None:
        features = ['River', 'T', 'EC', 'DOpercent', 'DO', 'pH', 'ORP', 'Turbidity',
                    'TSS', 'E-coli_4dilutions']

    features = check_attributes(features, df.columns.tolist())
    df = df[features]

    # River is not a representative name
    df = df.rename(columns={"River": "station_name"})

    if st:
        if isinstance(en, int):
            assert isinstance(en, int)
            df = df.iloc[st:en]
        else:
            df = df.loc[st:en]

    return df