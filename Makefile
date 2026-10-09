CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -Wall -Wextra
LDLIBS  = -lm

all: simulate

simulate: src/simulate.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

data: simulate
	mkdir -p data
	./simulate data

figures: data
	jupyter nbconvert --to notebook --execute --inplace notebooks/plot_figures.ipynb

clean:
	rm -f simulate

.PHONY: all data figures clean
