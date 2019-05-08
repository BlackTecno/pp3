CC=gcc
SRCS= $(wildcard *.c)
OBJS = $(SRCS:.c=.exe)
LIBS = -fopenmp


all:$(OBJS)

%.exe:%.c
	$(CC) $^ $(LIBS) -o $@

clean:
	rm -f ./*.o ./*.exe
