# The compilers to use
CC = g++

# C++ Compiler flags
CXXFLAGS = -Wall -g

# Include and Library directories
LIB = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs


all: run

run: clean
	$(CC) $(LIB) $(CXXFLAGS) main.cc -o run

clean:
	rm run
