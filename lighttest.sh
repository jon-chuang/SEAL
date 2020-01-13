export CXX=/usr/local/bin/syclcc-clang
export SEAL_USE_CXX17=FALSE
#rm -r native/build
#mkdir native/build
cd native/build
mkdir src
cd src
$HOME/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -DSEAL_USE_CXX17=$SEAL_USE_CXX17 ../../src
make -j8
mkdir ../examples
cd ../examples
$HOME/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -DSEAL_USE_CXX17=$SEAL_USE_CXX17 ../../examples
make -j8
mkdir ../tests
cd ../tests
$HOME/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -DSEAL_USE_CXX17=$SEAL_USE_CXX17 ../../tests
make -j8
cd ../../..
./native/bin/sealtest
