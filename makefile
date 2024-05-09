CC=mpicc
DEBUGFLAGS=-DDEBUG 
CFLAGS=-O3 -Wextra -Wall

all: gepp gepp_u

gepp: gepp.c
	$(CC) $(CFLAGS) -o gepp gepp.c

gepp_u: gepp_unrolling.c
	$(CC) $(CFLAGS) -o gepp_u gepp_unrolling.c

clean:
	rm -f gepp example gepp_d omp

