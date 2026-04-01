# examples/fortran/Makefile
REPO_ROOT    := $(abspath ../..)
LIB          := $(REPO_ROOT)/target/release
PUDDIN_MOD   := $(REPO_ROOT)/bindings/julia/include/puddin_binary.f90

example: example.f90 $(PUDDIN_MOD)
	$(FC) $(PUDDIN_MOD) $< -L$(LIB) -lpuddin_julia -Wl,-rpath,$(LIB) -o $@

.PHONY: clean
clean:
	rm -f example *.mod
