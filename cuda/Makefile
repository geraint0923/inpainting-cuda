CC = nvcc
CXXFLAGS = -std=c++11 -g -O2
LDFLAGS = `pkg-config --libs opencv`

all: inpainting

inpainting: inpainting.o cuda_inpainting.o
#	g++ -std=c++11 -g -O2 -o $@ $< `pkg-config --libs opencv`
	nvcc -o $@ inpainting.o cuda_inpainting.o $(LDFLAGS)

inpainting.o: inpainting.cpp cuda_inpainting.h

cuda_inpainting.o: cuda_inpainting.cpp cuda_inpainting.h
	nvcc $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf inpainting *.o
