CC=mpicc
DEBUGFLAGS=-DDEBUG

all: gepp

gepp: gepp.c
	$(CC) $(CFLAGS) -o gepp gepp.c $(OMPFLAGS)

debug: gepp.c
	$(CC) $(CFLAGS) $(DEBUGFLAGS) -o gepp_d gepp.c $(OMPFLAGS)

clean:
	rm -f gepp example gepp_d omp

