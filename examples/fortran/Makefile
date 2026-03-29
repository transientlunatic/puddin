# examples/fortran/Makefile
REPO_ROOT := $(abspath ../..)
LIB       := $(REPO_ROOT)/target/release

example: example.f90
	$(FC) $< -L$(LIB) -lpuddin_julia -Wl,-rpath,$(LIB) -o $@

.PHONY: clean
clean:
	rm -f example *.mod
