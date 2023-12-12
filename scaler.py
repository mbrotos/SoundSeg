import numpy as np
from sklearn.preprocessing import RobustScaler

def spec_transpose(spec, normalization):
    return np.transpose(spec, (0, 2, 1, 3)) if normalization == "frequency" else spec

def normalize(mix, vocal, normalization="frequency", quantile_scaler=False, q_min=25.0, q_max=75.0):
    if quantile_scaler:
        mix, vocal = spec_transpose(mix, normalization), spec_transpose(vocal, normalization)
        mix_scaled = np.zeros_like(mix)
        vocal_scaled = np.zeros_like(vocal)
        scaler_params = []
        for i in range(len(mix)):
            scaler = RobustScaler(quantile_range=(q_min, q_max)).fit(mix[i, :,:, 0])
            mix_scaled[i, :,:, 0] = scaler.transform(mix[i, :,:, 0])
            vocal_scaled[i, :,:, 0] = scaler.transform(vocal[i, :,:, 0])
            cur_params = {
                'params': scaler.get_params(deep=True),
                'center_': scaler.center_,
                'scale_': scaler.scale_,
                'n_features_in_': scaler.n_features_in_,
            }
            scaler_params.append(cur_params)
        return spec_transpose(mix_scaled, normalization), spec_transpose(vocal_scaled, normalization), scaler_params
    else:
        axis = (1, 3) if normalization == "time" else (2, 3)
        maxes = np.max(mix, axis=axis)
        divisor = (
            maxes[:, None, :, None]
            if normalization == "time"
            else maxes[:, :, None, None]
        )
        np.divide(mix, divisor, out=mix, where=divisor != 0)
        np.divide(vocal, divisor, out=vocal, where=divisor != 0)
        return mix, vocal, divisor
    
def denormalize(spec, scaler_params, normalization="frequency", quantile_scaler=False):
    if quantile_scaler:
        spec = spec_transpose(spec, normalization)
        spec_scaled = np.zeros_like(spec)
        for i in range(len(spec)):
            scaler = RobustScaler()
            scaler.set_params(**scaler_params[i]['params'])
            scaler.center_ = scaler_params[i]['center_']
            scaler.scale_ = scaler_params[i]['scale_']
            scaler.n_features_in_ = scaler_params[i]['n_features_in_']
            spec_scaled[i, :,:, 0] = scaler.inverse_transform(spec[i, :,:, 0])
        return spec_transpose(spec_scaled, normalization)
    else:
        return spec * scaler_params
