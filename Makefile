# Root shim: forwards to src/Makefile so `make` at the repo root keeps working.
# Real build rules live in src/Makefile; binaries land in bin/.

.PHONY: all clean

all clean:
	$(MAKE) -C src $@
