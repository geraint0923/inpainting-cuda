CC = nvcc
CXXFLAGS = -std=c++11 -g -O2
LDFLAGS = `pkg-config --libs opencv`

all: inpainting

inpainting: inpainting.o inpainting_cuda.o
#	g++ -std=c++11 -g -O2 -o $@ $< `pkg-config --libs opencv`
	$(CC) -o $@ inpainting.o inpainting_cuda.o $(LDFLAGS)

inpainting.o: inpainting.cpp inpainting_cuda.h

inpainting_cuda.o: inpainting_cuda.cu inpainting_cuda.h
	$(CC) $(CXXFLAGS) -c $<

.PHONY: clean

clean:
	rm -rf inpainting *.o
