all:
	mkdir -p bin
	nvcc -Xptxas -v -lineinfo -O3 -DFORMATTING src/main.cu -o bin/cuda-checkers

clean:
	rm -rf bin/*
