HOST_COMPILER  = g++
NVCC = nvcc

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -lineinfo
NVCC_DBG       = -O3 -g -use_fast_math

# --device-debug                                  (-G)                            
#         Generate debug information for device code. If --dopt is not specified, then
#         turns off all optimizations. Don't use for profiling; use -lineinfo instead.

# --generate-line-info                            (-lineinfo)                     
#         Generate line-number information for device code.

ARCH=sm_86

NVCCFLAGS      = $(NVCC_DBG) -arch=$(ARCH) -std=c++20
#GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) -o cudart cudart.o

cudart.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) -o cudart.o -c main.cu

out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: cudart
	nvprof ./cudart > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

test: cudart
	./cudart > out.ppm
	pytest test_compare_ppm.py  

clean:
	rm -f cudart cudart.o
