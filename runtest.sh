cd native/src
make clean
make -j8
cd ../tests
make clean
make -j8
cd ../..
./native/bin/sealtest
