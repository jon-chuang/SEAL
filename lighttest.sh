source ~/.bashrc
export CXX=/usr/local/bin/syclcc-clang
export SEAL_USE_CXX17=FALSE
#rm -r native/build
#mkdir native/build
cd native/build
#mkdir src
cd src
echo `which cmake`
make -j8
cd ../examples
make -j8
cd ../tests
make -j8
cd ../../..
#./native/bin/sealtest
