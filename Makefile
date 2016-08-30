# The compilers to use
CC = g++

# C++ Compiler flags
CXXFLAGS = -c -Wall -g

# Include and Library directories
LIB = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs


all: run

run: clean
	$(CC) $(LIB) main.cc -o run

clean:
	rm run
