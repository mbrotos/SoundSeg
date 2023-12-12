import unittest
import numpy as np
from scaler import spec_transpose, normalize, denormalize
from augmentations import consecutive_oversample, blackout, random_oversample

class TestAudioProcessing(unittest.TestCase):

    def setUp(self):
        self.num_samples = np.random.randint(2, 100)
        self.mix = np.random.rand(self.num_samples, 512, 128, 1)  # Dummy spectrogram data
        self.vocal = np.random.rand(self.num_samples, 512, 128, 1)

    def test_spec_transpose(self):
        # Test for frequency normalization
        transposed = spec_transpose(self.mix, "frequency")
        self.assertEqual(transposed.shape, (self.num_samples, 128, 512, 1))

        # Test for no transpose
        not_transposed = spec_transpose(self.mix, "time")
        self.assertEqual(not_transposed.shape, self.mix.shape)

    def test_normalize(self):
        # Test without quantile scaler
        normalized_mix_freq, normalized_vocal_freq, normalized_vocal_freq_scaler = normalize(self.mix, self.vocal, normalization='frequency', quantile_scaler=False)
        self.assertEqual(normalized_mix_freq.shape, self.mix.shape)
        self.assertEqual(normalized_vocal_freq.shape, self.vocal.shape)
        self.assertEqual(normalized_vocal_freq_scaler.shape, (self.num_samples, 512, 1, 1))
        
        
        normalized_mix_time, normalized_vocal_time, normalized_vocal_time_scaler = normalize(self.mix, self.vocal, normalization='time', quantile_scaler=False)
        self.assertEqual(normalized_mix_time.shape, self.mix.shape)
        self.assertEqual(normalized_vocal_time.shape, self.vocal.shape)
        self.assertEqual(normalized_vocal_time_scaler.shape, (self.num_samples, 1, 128, 1))

        # Test with quantile scaler
        normalized_mix_freq, normalized_vocal_freq, scaler_params = normalize(self.mix, self.vocal, normalization='frequency', quantile_scaler=True)
        self.assertEqual(normalized_mix_freq.shape, self.mix.shape)
        self.assertEqual(normalized_vocal_freq.shape, self.vocal.shape)
        self.assertEqual(len(scaler_params), self.num_samples)
        self.assertEqual(scaler_params[0]['n_features_in_'], 512)
        
        normalized_mix_time, normalized_vocal_time, scaler_params = normalize(self.mix, self.vocal, normalization='time', quantile_scaler=True)
        self.assertEqual(normalized_mix_time.shape, self.mix.shape)
        self.assertEqual(normalized_vocal_time.shape, self.vocal.shape)
        self.assertEqual(len(scaler_params), self.num_samples)
        self.assertEqual(scaler_params[0]['n_features_in_'], 128)

    def test_denormalize(self):
        # Test denormalization process
        normalized_mix_freq, _, normalized_mix_freq_scaler = normalize(self.mix, self.vocal, normalization='frequency', quantile_scaler=False)
        denormalized_mix_freq = denormalize(normalized_mix_freq, normalized_mix_freq_scaler, normalization='frequency', quantile_scaler=False)
        self.assertEqual(denormalized_mix_freq.shape, self.mix.shape)
        self.assertTrue(np.allclose(denormalized_mix_freq, self.mix, atol=5e-1))
        
        normalized_mix_time, _, normalized_mix_time_scaler = normalize(self.mix, self.vocal, normalization='time', quantile_scaler=False)
        denormalized_mix_time = denormalize(normalized_mix_time, normalized_mix_time_scaler, normalization='time', quantile_scaler=False)
        self.assertEqual(denormalized_mix_time.shape, self.mix.shape)
        self.assertTrue(np.allclose(denormalized_mix_time, self.mix, atol=5e-1))
        
        # Test denormalization process with quantile scaler
        normalized_mix_freq, _, scaler_params = normalize(self.mix, self.vocal, normalization='frequency', quantile_scaler=True)
        denormalized_mix_freq = denormalize(normalized_mix_freq, scaler_params, normalization='frequency', quantile_scaler=True)
        self.assertEqual(denormalized_mix_freq.shape, self.mix.shape)
        self.assertTrue(np.allclose(denormalized_mix_freq, self.mix))
        
        normalized_mix_time, _, scaler_params = normalize(self.mix, self.vocal, normalization='time', quantile_scaler=True)
        denormalized_mix_time = denormalize(normalized_mix_time, scaler_params, normalization='time', quantile_scaler=True)
        self.assertEqual(denormalized_mix_time.shape, self.mix.shape)
        self.assertTrue(np.allclose(denormalized_mix_time, self.mix))
        
    def test_consecutive_oversample(self):
        # Test consecutive oversampling
        mix_consec, vocal_consec = consecutive_oversample(self.mix, self.vocal)
        self.assertEqual(mix_consec.shape, (self.num_samples - 1, 512, 128, 1))
        self.assertEqual(vocal_consec.shape, (self.num_samples - 1, 512, 128, 1))
        
    def test_random_oversample(self):
        # Test random oversampling
        mix_rand, vocal_rand = random_oversample(self.mix, self.vocal)
        self.assertEqual(mix_rand.shape, (self.num_samples - 1, 512, 128, 1))
        self.assertEqual(vocal_rand.shape, (self.num_samples - 1, 512, 128, 1))
        
    def test_blackout(self):
        # Test blackout
        mix_blackout, vocal_blackout = blackout(self.mix, self.vocal)
        self.assertEqual(mix_blackout.shape, self.mix.shape)
        self.assertEqual(vocal_blackout.shape, self.vocal.shape)
        

if __name__ == '__main__':
    unittest.main()
