CC=mpicc
DEBUGFLAGS=-DDEBUG 
CFLAGS=-Wextra -Wall

all: gepp gepp_u

gepp: gepp.c
	$(CC) $(CFLAGS) -o gepp gepp.c

debug: gepp.c
	$(CC) -g $(CFLAGS) -o gepp_d gepp.c

gepp_u: gepp_unrolling.c
	$(CC) $(CFLAGS) -o gepp_u gepp_unrolling.c

clean:
	rm -f gepp gepp_d

