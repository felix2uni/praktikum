all:
	$(MAKE) -C vXXX
	$(MAKE) -C V351
	$(MAKE) -C V103
	$(MAKE) -C V303

clean:
	$(MAKE) -C vXXX clean
	$(MAKE) -C V351 clean
	$(MAKE) -C V103 clean
	$(MAKE) -C V303 clean

.PHONY: all clean
