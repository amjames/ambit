set(TENSOR_HEADERS
        ${PROJECT_SOURCE_DIR}/include/ambit/print.h
        ${PROJECT_SOURCE_DIR}/include/ambit/tensor.h
        ${PROJECT_SOURCE_DIR}/include/ambit/timer.h
        ${PROJECT_SOURCE_DIR}/include/ambit/blocked_tensor.h
        ${PROJECT_SOURCE_DIR}/include/ambit/sym_blocked_tensor.h
        ${PROJECT_SOURCE_DIR}/include/ambit/common_types.h
        ${PROJECT_SOURCE_DIR}/include/ambit/settings.h

        ../include/ambit/io/hdf5.h
        ../include/ambit/io/hdf5/attribute.h
        ../include/ambit/io/hdf5/dataset.h
        ../include/ambit/io/hdf5/dataspace.h
        ../include/ambit/io/hdf5/file.h
        ../include/ambit/io/hdf5/group.h
        ../include/ambit/io/hdf5/location.h
        ../include/ambit/io/psi4/file.h
        ../include/ambit/io/psi4/io.h
        ../include/ambit/io/psi4/iwl.h
        ../include/ambit/io/psi4/manager.h

        ${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/io.h

        math/math.h

        tensor/core/core.h
        tensor/disk/disk.h
        tensor/indices.h
        tensor/globals.h
        tensor/macros.h
        tensor/tensorimpl.h
        )

set(TENSOR_SOURCES
        helpers/psi4/io.cc

        io/hdf5/attribute.cc
        io/hdf5/dataset.cc
        io/hdf5/dataspace.cc
        io/hdf5/file.cc
        io/hdf5/group.cc
        io/hdf5/location.cc

        io/psi4/file.cc
        io/psi4/iwl.cc
        io/psi4/manager.cc

        math/blas.cc
        math/lapack.cc

        tensor/core/core.cc
        tensor/disk/disk.cc

        tensor/indices.cc
        tensor/globals.cc
        tensor/labeled_tensor.cc
        tensor/print.cc
        tensor/slice.cc
        tensor/sliced_tensor.cc
        tensor/tensor.cc
        tensor/tensorimpl.cc
        tensor/timer.cc

        blocked_tensor/blocked_tensor.cc
        blocked_tensor/sym_blocked_tensor.cc
        )

# if we have MPI and Cyclops is enabled
if (ENABLE_CYCLOPS)
    list(APPEND TENSOR_HEADERS tensor/cyclops/cyclops.h)
    list(APPEND TENSOR_SOURCES tensor/cyclops/cyclops.cc)
endif ()

if (ENABLE_PSI4)
    list(APPEND TENSOR_HEADERS
            ${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/integrals.h
            ${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/convert.h)
    list(APPEND TENSOR_SOURCES
            helpers/psi4/integrals.cc
            helpers/psi4/convert.cc)
endif ()

list(SORT TENSOR_SOURCES)

# build library
if (NOT STATIC_ONLY)
    add_library(ambit-shared SHARED ${TENSOR_SOURCES} ${TENSOR_HEADERS})
    set_target_properties (ambit-shared PROPERTIES CLEAN_DIRECT_OUTPUT 1 OUTPUT_NAME "ambit")
    install(TARGETS ambit-shared LIBRARY DESTINATION lib)
endif (NOT STATIC_ONLY)

add_library(ambit-static STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})
set_target_properties (ambit-static PROPERTIES CLEAN_DIRECT_OUTPUT 1 OUTPUT_NAME "ambit")
install(TARGETS ambit-static ARCHIVE DESTINATION lib)

if (ENABLE_CYCLOPS)
    if (NOT STATIC_ONLY)
        target_link_libraries(ambit-shared ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
    endif()
    target_link_libraries(ambit-static ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
endif ()

if (NOT STATIC_ONLY)
    target_link_libraries(ambit-shared
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${Boost_LIBRARIES}
        ${HDF5_LIBRARIES}
        )
endif()
target_link_libraries(ambit-static
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${Boost_LIBRARIES}
        ${HDF5_LIBRARIES}
        )

if (NOT STATIC_ONLY)
    add_library(ambit ALIAS ambit-shared)
else()
    add_library(ambit ALIAS ambit-static)
endif()

if (ENABLE_PYTHON)
    add_subdirectory(python)
endif ()

