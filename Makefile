# The compilers to use
CC = nvcc

# C++ Compiler flags
CXXFLAGS = -O3 -std=c++11

# Include and Library directories
LIB = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

all: run

run: clean
	$(CC) $(LIB) $(CXXFLAGS) main.cu -o run

clean:
	rm run
