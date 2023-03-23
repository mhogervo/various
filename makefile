CC=c++
CFLAGS=-c -std=c++17 -O2 -Wall -Wextra -Wunused -pedantic
LFLAGS=-Wl

all: rand

rand: rand.o random_walks.o
	$(CC) $(LFLAGS) rand.o random_walks.o -o rand

rand.o: rand.cpp random_walks.h
	$(CC) $(CFLAGS) rand.cpp

random_walks.o: random_walks.cpp random_walks.h
	$(CC) $(CFLAGS) random_walks.cpp

clean:
	rm -rf *.o *.h~ *.cpp~ rand
