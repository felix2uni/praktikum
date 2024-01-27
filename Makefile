all:
	$(MAKE) -C vXXX
	$(MAKE) -C V351
	$(MAKE) -C V103
	$(MAKE) -C V303
	$(MAKE) -C V207
	$(MAKE) -C V106
	$(MAKE) -C V206

clean:
	$(MAKE) -C vXXX clean
	$(MAKE) -C V351 clean
	$(MAKE) -C V103 clean
	$(MAKE) -C V303 clean
	$(MAKE) -C V207 clean
	$(MAKE) -C V206 clean

.PHONY: all clean
