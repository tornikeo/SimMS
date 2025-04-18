import pytest
import numba
from numba import cuda
from simms.similarity.spectrum_similarity_functions import cosine_kernel
from simms.similarity import CudaCosineGreedy
from matchms.filtering import reduce_to_number_of_peaks
import numpy as np
import time
from itertools import product, pairwise


# @pytest.fixture(scope="session")
# def gnps_tensors(gnps: list):
#   mz = np.load('data/gnps-mz.npy')
#   int_ = np.load('data/gnps-int.npy')
#   specra_lens = np.load('data/gnps-spectra_lens.npy')
#   return mz, int, specra_lens
@pytest.fixture
def my_spectra(gnps):
    spectra = [reduce_to_number_of_peaks(sp, n_max=1024) for sp in gnps]
    spectra = [sp for sp in spectra if sp is not None]
    return spectra

def test_cosine_kernel_perf(my_spectra: list):
  my_spectra = my_spectra[:len(my_spectra)//2]
  cosine_greedy = CudaCosineGreedy(batch_size=2048, verbose=True)
  start_time = time.time()
  cosine_greedy.matrix(my_spectra, my_spectra)
  end_time = time.time()
  print(f"Time taken for cosine_greedy.matrix call: {end_time - start_time:.2f} seconds")

  # mz, int_, spectra_lens = gnps_tensors
  # kernel = cosine_kernel(
  #   tolerance=0.1, 
  #   shift=0, 
  #   mz_power=0, int_power=1, 
  #   match_limit=1024, n_max_peaks=2048
  # )
  # batch_size = 2048
  # for i in range(len(mz)):
  #   for j in range(len(mz)):
  #     mz_i, mz_j = mz[i:i+batch_size], mz[j:j+batch_size]

  #     metadata = cuda.device_array()
  #     rspec, qspec = cuda.to_device(mz_i), cuda.to_device(mz_j)
  #     kernel(rspec, qspec, metadata, out)
