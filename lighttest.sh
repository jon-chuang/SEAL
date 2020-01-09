cd native/src
#$HOME/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -DSEAL_USE_CXX17=FALSE .
#make clean
make -j8
cd ../tests
#$HOME/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -DSEAL_USE_CXX17=FALSE .
#make clean
make -j8
cd ../..
./native/bin/sealtest
