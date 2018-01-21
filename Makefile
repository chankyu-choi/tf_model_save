#!/bin/bash

INCLUDE_PATH = tensorflow/include
LIBRARY_PATH = tensorflow/lib
LIBRARIES = -ltensorflow

CC = g++

SOURCE = main.cc
TARGET = main

all : $(TARGET)

$(TARGET) : $(SOURCE)
	$(CC) -o $@ $^ -I$(INCLUDE_PATH) -L$(LIBRARY_PATH) $(LIBRARIES)

clean :
	rm -rf *.o
	rm -rf $(TARGET)
