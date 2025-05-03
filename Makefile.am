# Define subdirectories
SUBDIRS = CellGrids

# Define the library to be built
pkglib_LTLIBRARIES = HwSampling.la

# Specify the library version information
HwSampling_la_LDFLAGS = $(AM_LDFLAGS) -module -version-info 8:0:0

# Specify the source files for the library
HwSampling_la_SOURCES = \
    GeneralStatistics.h GeneralStatistics.cc \
    MultiIterationStatistics.h MultiIterationStatistics.cc \
    BinSampler.h BinSampler.cc \
    GeneralSampler.h GeneralSampler.cc \
    MonacoSampler.h MonacoSampler.cc \
    Remapper.h Remapper.cc \
    PythonSampler.h PythonSampler.cc \
    exsample/cell.icc \
    exsample/generator.h \
    exsample/linear_interpolator.icc \
    exsample/config.h \
    exsample/selectors.h \
    exsample/adaption_info.h \
    exsample/selectors.icc \
    exsample/adaption_info.icc \
    exsample/exponential_generator.h \
    exsample/exponential_generator.icc \
    exsample/generator.icc \
    exsample/statistics.h \
    exsample/linear_interpolator.h \
    exsample/cell.h \
    exsample/utility.h \
    exsample/binary_tree.h \
    exsample/statistics.icc

# Specify additional libraries to link against
HwSampling_la_LIBADD = \
    CellGrids/libHwCellGrids.la \
    -lpython3.10

# Specify the library paths for linking
HwSampling_la_LDFLAGS += -L/home/finn/.pyenv/versions/3.10.16/lib

# Specify the include paths for Pybind11 and Python
HwSampling_la_CPPFLAGS = $(AM_CPPFLAGS) \
                        -I/home/finn/.pyenv/versions/3.10.16/include/python3.10 \
                        -I/home/finn/.pyenv/versions/madnis/lib/python3.10/site-packages/pybind11/include

# Specify the C++ compiler flags
HwSampling_la_CXXFLAGS = $(AM_CXXFLAGS)
