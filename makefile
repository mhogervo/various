CC=c++
CFLAGS=-c -Wall -std=c++20
LFLAGS=

all: rand

rand: rand.o random_walks.o
	$(CC) $(LFLAGS) rand.o random_walks.o -o rand

rand.o: rand.cpp random_walks.h
	$(CC) $(CFLAGS) rand.cpp

random_walks.o: random_walks.cpp random_walks.h
	$(CC) $(CFLAGS) random_walks.cpp

clean:
	rm -rf *.o *.h~ rand
