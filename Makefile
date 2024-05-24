# mkae a wroking makefile for oneapi.cpp

CC = icpx -fsycl
TBIN = oneapi
ABIN = a3 

all: $(TBIN) $(ABIN)

$(TBIN): oneapi.cpp
	$(CC) -o $@ $<

$(ABIN): a3.cpp
	$(CC) -o $@ $<

clean:
	rm -f $(TBIN) $(ABIN)