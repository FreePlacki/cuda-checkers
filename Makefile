all:
	mkdir -p bin
	nvcc -O3 -DFORMATTING src/main.cu -lcurand -o bin/cuda-checkers

clean:
	rm -rf bin/*
