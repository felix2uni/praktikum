all:
	$(MAKE) -C vXXX
	$(MAKE) -C V351
	$(MAKE) -C V353
	$(MAKE) -C V103

clean:
	$(MAKE) -C vXXX clean
	$(MAKE) -C V351 clean
	$(MAKE) -C V353 clean
	$(MAKE) -C V103 clean
V351:
	$(MAKE) -C V351

.PHONY: all clean
