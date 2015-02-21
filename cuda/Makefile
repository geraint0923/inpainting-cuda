all: inpainting

inpainting: inpainting.cpp
	g++ -std=c++11 -g -O2 -o $@ $< `pkg-config --libs opencv`

.PHONY: clean

clean:
	rm -rf inpainting
