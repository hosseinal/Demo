# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-src"
  "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-build"
  "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-subbuild/metis-populate-prefix"
  "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-subbuild/metis-populate-prefix/tmp"
  "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-subbuild/metis-populate-prefix/src/metis-populate-stamp"
  "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-subbuild/metis-populate-prefix/src"
  "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-subbuild/metis-populate-prefix/src/metis-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-subbuild/metis-populate-prefix/src/metis-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/albakrih/CLionProjects/Demo/cmake-build-debug/_deps/metis-subbuild/metis-populate-prefix/src/metis-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
