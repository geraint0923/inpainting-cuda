CC = nvcc
CXXFLAGS = -std=c++11 -g -O0
LDFLAGS = `pkg-config --libs opencv`

all: inpainting

inpainting: inpainting.o cuda_inpainting.o
#	g++ -std=c++11 -g -O2 -o $@ $< `pkg-config --libs opencv`
	nvcc -o $@ inpainting.o cuda_inpainting.o $(LDFLAGS)

inpainting.o: inpainting.cpp cuda_inpainting.h

cuda_inpainting.o: cuda_inpainting.cu cuda_inpainting.h
	nvcc $(CXXFLAGS) -G -c $< -o $@

.PHONY: clean

clean:
	rm -rf inpainting *.o
