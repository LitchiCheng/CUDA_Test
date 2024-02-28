# CUDA_Test

## compile 
`nvcc -o calcArrayByCpuOrCuda calcArrayByCpuOrCuda.cu`

## result

|bytes|cuda cost|cpu cost|
|:-----|:-----|:-----|
|65536|0.069796|0.002085|
|262144|0.068762|0.008604|
|67108864|0.359864|2.18625|