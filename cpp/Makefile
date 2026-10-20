CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pthread

sim: sim.cpp rm.hpp rpa.hpp polar.hpp
	$(CXX) $(CXXFLAGS) -o $@ sim.cpp

clean:
	rm -f sim

.PHONY: clean
