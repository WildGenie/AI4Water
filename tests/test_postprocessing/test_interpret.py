import time
import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.datasets import busan_beach
from ai4water.postprocessing import Interpret
from ai4water import InputAttentionModel, DualAttentionModel

default_model = {
    'layers': {
        "Dense_0": {'units': 64, 'activation': 'relu'},
        "Flatten": {},
        "Dense_3": 1,
    }
}


examples = 2000
ins = 5
outs = 1
in_cols = ['input_'+str(i) for i in range(5)]
out_cols = ['output']
cols=  in_cols + out_cols
data1 = pd.DataFrame(np.arange(int(examples*len(cols))).reshape(-1, examples).transpose(),
                    columns=cols,
                    index=pd.date_range("20110101", periods=examples,
                                        freq="D"))


beach_data = busan_beach()
input_features=beach_data.columns.tolist()[0:-1]
output_features=beach_data.columns.tolist()[-1:]


def build_model(**kwargs):

    model = Model(
        verbosity=0,
        batch_size=16,
        epochs=1,
        **kwargs
    )

    return model


def da_lstm_model(data,
                  **kwargs):

    model = DualAttentionModel(
        split_random=True,
        verbosity=0,
        **kwargs
    )

    _ = model.fit(data=data)

    return model


def ia_lstm_model(**kwargs):

    model = InputAttentionModel(verbosity=0, **kwargs)

    _ = model.fit(data=beach_data)

    return model


def lstm_with_SelfAtten(**kwargs):
    return


def tft_model(**kwargs):
    num_encoder_steps = 30
    params = {
        'total_time_steps': num_encoder_steps,  # 8 * 24,
        'num_encoder_steps': num_encoder_steps,  # 7 * 24,
        'num_inputs': 7,
        'category_counts': [],
        'input_obs_loc': [],  # leave empty if not available
        'static_input_loc': [],  # if not static inputs, leave this empty
        'known_regular_inputs': [0, 1, 2, 3, 4, 5, 6],
        'known_categorical_inputs': [],  # leave empty if not applicable
        'hidden_units': 8,
        'dropout_rate': 0.1,
        'num_heads': 4,
        'use_cudnn': False,
        'future_inputs': False,
        'return_sequences': True,
    }

    output_size = 1
    layers = {
        "Input": {"config": {"shape": (params['total_time_steps'],
                                       params['num_inputs']),
                             'name': "Model_Input"}},
        "TemporalFusionTransformer": {"config": params},
        "lambda": {
            "config": tf.keras.layers.Lambda(lambda _x: _x[Ellipsis, -1, :])},
        "Dense": {"config": {"units": output_size * 1}},
    }

    model = Model(model={'layers': layers},
                  input_features=['et_morton_point_SILO',
                                  'precipitation_AWAP',
                                  'tmax_AWAP',
                                  'tmin_AWAP',
                                  'vprp_AWAP',
                                  'rh_tmax_SILO',
                                  'rh_tmin_SILO'
                                  ],
                  output_features=['streamflow_MLd_inclInfilled'],
                  dataset_args={'st': '19700101', 'en': '20141231',
                                'stations': '224206'},
                  ts_args={'lookback':num_encoder_steps},
                  epochs=2,
                  verbosity=0)
    return model


class TestInterpret(unittest.TestCase):

    def test_plot_feature_importance(self):

        time.sleep(1)
        model = build_model(
            ts_args={'lookback':7},
            input_features=in_cols,
            output_features=out_cols,
            model=default_model
        )

        model.fit(data=data1.astype(np.float32))

        Interpret(model).plot_feature_importance(np.random.randint(1, 10, 5))

        return

    def test_da_interpret(self):
        m = da_lstm_model(data='CAMELS_AUS',
                          ts_args={'lookback': 14},
                          input_features = ['precipitation_AWAP',
                          'evap_pan_SILO'],
        output_features = ['streamflow_MLd_inclInfilled'],
        intervals =  [("20000101", "20011231")],
        dataset_args = {'stations': 1},)
        m.interpret(show=False)
        return

    def test_da_without_prevy_interpret(self):
        m = da_lstm_model(teacher_forcing=False, drop_remainder=True,
                          data=beach_data,
                          input_features=input_features,
                          batch_size=8,
                          ts_args={'lookback':14},
                          output_features=output_features)
        x,y = m.training_data()


        m.interpret(data='training', show=False)
        return

    def test_ia_interpret(self):
        m = ia_lstm_model(input_features=input_features,
                          ts_args={'lookback': 14},
                          output_features=output_features)
        m.interpret(show=False)
        return

    def test_ml(self):
        time.sleep(1)
        for m in ['XGBRegressor', 'RandomForestRegressor',
                  'GradientBoostingRegressor', 'LinearRegression']:

            model = build_model(model=m)
            model.fit(data=beach_data)
            model.interpret()
        return

    def test_tft(self):

        model = tft_model()
        model.fit(data='CAMELS_AUS')

        i = Interpret(model)

        i.interpret_example_tft(0, data='test')

        i.interpret_tft(data='test')
        return

    def test_xgb_f_imp_comparison(self):
        model = Model(model="XGBRegressor")
        model.fit(data=busan_beach(inputs=["tide_cm", "rel_hum"]))
        interpreter = Interpret(model)
        interpreter.compare_xgb_f_imp()
        return


if __name__ == "__main__":
    unittest.main()